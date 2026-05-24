"""
03_spark_streaming_job.py  —  Spark Structured Streaming pipeline for sepsis prediction.

Architecture (Kafka mode, default):
  Kafka topic mimic-icu-stream
    → readStream("kafka") + from_json parse
    → watermark + per-patient sliding-window aggregation
      (pivot embedded as conditional aggregation — pivot() is unsupported on streams)
    → stream-static join with cohort.parquet for Sepsis-3 "next 6h" labels
    → Parquet sink  (data/stream_output/)
    → latency log   (data/processed/latency_log.csv)  [Kafka path only]

Architecture (CSV fallback, --source csv):
  CSV files (data/stream_input/)  →  identical aggregation  →  Parquet sink

Run (from project root):
  # 1. Start Kafka
  docker compose -f docker/docker-compose.yml up -d

  # 2. Run producer (after Kafka is healthy)
  .\.venv\Scripts\python src/03a_kafka_producer.py

  # 3. Run Spark job (Kafka source, default)
  $env:PYSPARK_PYTHON = ".\.venv\Scripts\python.exe"
  spark-submit.cmd --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 ^
      src/03_spark_streaming_job.py --source kafka

  # Or via Python (downloads Kafka package automatically):
  .\.venv\Scripts\python src/03_spark_streaming_job.py --source kafka

  # CSV fallback (original behaviour):
  .\.venv\Scripts\python src/03_spark_streaming_job.py --source csv
"""

import argparse
import os
import sys
from pathlib import Path

# ── 1. Spark path bootstrap ───────────────────────────────────────────────────
SPARK_HOME = os.environ.get("SPARK_HOME")
if not SPARK_HOME:
    sys.exit(
        "ERROR: SPARK_HOME is not set.\n"
        "  PowerShell: $env:SPARK_HOME = 'E:\\Big Data\\spark'\n"
        "  Then re-run this script."
    )

_spark_home = Path(SPARK_HOME)

if not os.environ.get("HADOOP_HOME"):
    _candidate = _spark_home.parent / "hadoop"
    if _candidate.exists():
        os.environ["HADOOP_HOME"] = str(_candidate)

os.environ.setdefault("PYSPARK_PYTHON",        sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

try:
    import pyspark  # noqa: F401
except ImportError:
    _pyspark_zip = _spark_home / "python" / "lib" / "pyspark.zip"
    _py4j_zips   = sorted((_spark_home / "python" / "lib").glob("py4j-*.zip"))
    if not _pyspark_zip.exists():
        sys.exit(
            f"ERROR: pyspark.zip not found at {_pyspark_zip}.\n"
            "Ensure SPARK_HOME points to a valid Spark installation."
        )
    sys.path.insert(0, str(_pyspark_zip))
    if _py4j_zips:
        sys.path.insert(0, str(_py4j_zips[0]))

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQueryListener
from pyspark.sql.types import (
    DoubleType, LongType, StringType,
    StructField, StructType, TimestampType,
)

# ── 2. Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
STREAM_INPUT  = DATA_DIR / "stream_input"
STREAM_OUTPUT = DATA_DIR / "stream_output"
COHORT_PATH   = DATA_DIR / "processed" / "cohort.parquet"
WAREHOUSE_DIR = PROJECT_ROOT / "spark-warehouse"
LATENCY_LOG   = DATA_DIR / "processed" / "latency_log.csv"

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC             = "mimic-icu-stream"
# Maven artifact for the Kafka connector matching this Spark version
KAFKA_PACKAGE           = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"

SIGNALS: list[str] = [
    "bilirubin", "creatinine", "dbp", "heart_rate",
    "lactate", "map", "platelets", "resp_rate",
    "sbp", "spo2", "temperature", "wbc",
]

WINDOW_DURATION = "1 hour"
SLIDE_DURATION  = "30 minutes"
WATERMARK_DELAY = "1 hour"
LABEL_HORIZON_H = 6

FORWARD_FILL_ENABLED = False

# ── 3. Input schemas ──────────────────────────────────────────────────────────

# Schema for CSV source (original)
CSV_SCHEMA = StructType([
    StructField("icustay_id",   LongType(),      True),
    StructField("charttime",    TimestampType(), True),
    StructField("signal_name",  StringType(),    True),
    StructField("valuenum",     DoubleType(),    True),
    StructField("source_table", StringType(),    True),
])

# Schema for the JSON payload embedded in each Kafka message value
KAFKA_JSON_SCHEMA = StructType([
    StructField("icustay_id",            LongType(),   True),
    StructField("charttime",             StringType(), True),  # string → cast to Timestamp
    StructField("signal_name",           StringType(), True),
    StructField("valuenum",              DoubleType(), True),
    StructField("source_table",          StringType(), True),
    StructField("producer_timestamp_ms", LongType(),   True),
])

# ── 4. Window aggregation expressions ─────────────────────────────────────────

def _window_agg_exprs() -> list:
    """
    Per-signal window stats using conditional aggregation (pivot is banned on streams).
    Slope = OLS slope via covar_samp / var_samp; returns null for single-observation windows.
    72 columns: 12 signals × 6 stats (mean, stddev, min, max, last, slope).
    """
    exprs = []
    for sig in SIGNALS:
        val   = F.when(F.col("signal_name") == sig, F.col("valuenum"))
        t_unix = F.when(
            F.col("signal_name") == sig,
            F.unix_timestamp(F.col("charttime")).cast(DoubleType()),
        )
        exprs += [
            F.mean(val)                         .alias(f"{sig}_mean"),
            F.stddev(val)                       .alias(f"{sig}_stddev"),
            F.min(val)                          .alias(f"{sig}_min"),
            F.max(val)                          .alias(f"{sig}_max"),
            F.last(val, ignorenulls=True)       .alias(f"{sig}_last"),
            (F.covar_samp(t_unix, val) / F.var_samp(t_unix)).alias(f"{sig}_slope"),
        ]
    return exprs


# ── 5. Forward-fill (disabled by default) ─────────────────────────────────────

def _ffill_state_schema() -> str:
    return ", ".join(f"{s}_state double" for s in SIGNALS)


def _forward_fill_fn(key, pdf_iter, state):
    if state.exists:
        row  = state.get
        last = {s: getattr(row, f"{s}_state", None) for s in SIGNALS}
    else:
        last = {s: None for s in SIGNALS}

    out = []
    for pdf in pdf_iter:
        for sig in SIGNALS:
            col = f"{sig}_last"
            if col not in pdf.columns:
                continue
            if last[sig] is not None:
                pdf[col] = pdf[col].fillna(last[sig])
            nonnull = pdf[col].dropna()
            if len(nonnull):
                last[sig] = float(nonnull.iloc[-1])
        out.append(pdf)

    state.update(pd.DataFrame({f"{s}_state": [last[s]] for s in SIGNALS}))
    state.setTimeoutDuration(24 * 60 * 60 * 1000)

    if out:
        yield pd.concat(out, ignore_index=True)


# ── 6. Streaming progress listener ────────────────────────────────────────────

class _ProgressListener(StreamingQueryListener):
    def onQueryStarted(self, event):
        print(f"[Listener] Query started  id={event.id}")

    def onQueryProgress(self, event):
        p = event.progress
        print(
            f"[Listener] Batch {p.batchId:>4d}  "
            f"inputRows={p.numInputRows:,}  "
            f"processedRowsPerSecond={p.processedRowsPerSecond:.0f}"
        )

    def onQueryTerminated(self, event):
        print(
            f"[Listener] Query terminated  id={event.id}  "
            f"exception={event.exception or 'none'}"
        )


# ── 7. SparkSession ───────────────────────────────────────────────────────────

def _build_spark(source: str) -> SparkSession:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder
        .master("local[4]")
        .appName("sepsis-streaming")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR.as_posix())
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.sql.ansi.enabled", "false")
    )
    if source == "kafka":
        # Download Kafka connector from Maven Central (cached after first run).
        # If this version is unavailable, try replacing 4.1.1 with 4.0.0.
        builder = builder.config("spark.jars.packages", KAFKA_PACKAGE)
    return builder.getOrCreate()


# ── 8. Stream sources ─────────────────────────────────────────────────────────

def _build_csv_stream(spark: SparkSession):
    """Original CSV file-source (fallback)."""
    csv_files = list(STREAM_INPUT.glob("*.csv"))
    print(f"CSV source: {len(csv_files)} files in {STREAM_INPUT}")
    raw = (
        spark.readStream
        .format("csv")
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .option("pathGlobFilter", "*.csv")
        .schema(CSV_SCHEMA)
        .load(STREAM_INPUT.as_posix())
    )
    return raw


def _build_kafka_stream(spark: SparkSession):
    """
    Kafka source: reads from topic mimic-icu-stream, parses JSON payload,
    casts charttime string to Timestamp, and adds latency measurement columns:
      processing_timestamp_ms — wall-clock ms when Spark ingested this row
      latency_ms              — processing_timestamp_ms − producer_timestamp_ms
    """
    raw_kafka = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw_kafka
        .select(F.from_json(F.col("value").cast("string"), KAFKA_JSON_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("charttime",
                    F.to_timestamp(F.col("charttime"), "yyyy-MM-dd HH:mm:ss"))
        .withColumn("processing_timestamp_ms",
                    (F.unix_timestamp(F.current_timestamp()) * 1000).cast(LongType()))
        .withColumn("latency_ms",
                    F.col("processing_timestamp_ms") - F.col("producer_timestamp_ms"))
    )
    return parsed


# ── 9. Label join (stream-static) ─────────────────────────────────────────────

def _attach_labels(df, spark: SparkSession):
    cohort = (
        spark.read.parquet(COHORT_PATH.as_posix())
        .select("icustay_id", "sepsis_label", "sepsis_onset_time")
    )
    df = df.join(F.broadcast(cohort), on="icustay_id", how="left")

    horizon = F.expr(f"INTERVAL {LABEL_HORIZON_H} HOURS")
    df = df.withColumn(
        "label",
        F.when(
            (F.col("sepsis_label") == 1)
            & (F.col("sepsis_onset_time") > F.col("window_end_time"))
            & (F.col("sepsis_onset_time") <= F.col("window_end_time") + horizon),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    return df.drop("sepsis_label", "sepsis_onset_time")


# ── 10. Sink helpers ──────────────────────────────────────────────────────────

# Columns added for latency measurement (Kafka path only); dropped before Parquet write
# so the output schema is identical to the CSV path.
_LATENCY_COLS = ["avg_latency_ms", "min_latency_ms", "max_latency_ms", "n_events"]


def _make_foreachbatch(checkpoint_dir: Path):
    """
    Returns a foreachBatch function for the Kafka sink.
    Writes:
      • Parquet rows to STREAM_OUTPUT (latency cols dropped — same schema as CSV path)
      • Per-window latency stats to LATENCY_LOG (CSV, appended across batches)
    """
    STREAM_OUTPUT.mkdir(parents=True, exist_ok=True)
    # Delete latency log from any prior run so we start fresh
    if LATENCY_LOG.exists():
        LATENCY_LOG.unlink()

    def _write_batch(df, epoch_id: int) -> None:
        if df.rdd.isEmpty():
            return

        # ── Write latency stats ───────────────────────────────────────────────
        lat_cols_present = [c for c in _LATENCY_COLS if c in df.columns]
        if lat_cols_present:
            lat_pdf = df.select(
                "icustay_id", "window_end_time",
                *lat_cols_present
            ).toPandas()
            # throughput_per_window: events in 30-min slide window
            if "n_events" in lat_pdf.columns:
                lat_pdf["throughput_per_window"] = lat_pdf["n_events"]
            lat_pdf.to_csv(
                str(LATENCY_LOG),
                mode="a",
                header=not LATENCY_LOG.exists(),
                index=False,
            )

        # ── Write Parquet (drop latency cols to match CSV schema) ─────────────
        parquet_df = df.drop(*[c for c in _LATENCY_COLS if c in df.columns])
        parquet_df.write.mode("append").parquet(STREAM_OUTPUT.as_posix())

    return _write_batch


# ── 11. Post-run report ───────────────────────────────────────────────────────

def _print_report(spark: SparkSession) -> None:
    print("\n" + "=" * 62)
    print("POST-RUN STATISTICS")
    print("=" * 62)

    out = spark.read.parquet(STREAM_OUTPUT.as_posix())
    total = out.count()
    print(f"  Total output rows (windows) : {total:,}")

    if total == 0:
        print("  (no output — check watermark / input data)")
        return

    label_rows = out.groupBy("label").count().orderBy("label").collect()
    print("  Label distribution:")
    for r in label_rows:
        pct = r["count"] / total * 100
        print(f"    label={r['label']}  count={r['count']:,}  ({pct:.1f}%)")

    win_per_stay = out.groupBy("icustay_id").count()
    stats = win_per_stay.agg(
        F.min("count").alias("min"),
        F.max("count").alias("max"),
        F.mean("count").alias("mean"),
        F.percentile_approx("count", 0.5).alias("median"),
        F.count("*").alias("n_stays"),
    ).collect()[0]
    print(
        f"  Windows per icustay_id  "
        f"(n_stays={stats['n_stays']}):  "
        f"min={stats['min']}  max={stats['max']}  "
        f"mean={stats['mean']:.1f}  median={stats['median']}"
    )

    counts_list = [r["count"] for r in win_per_stay.collect()]
    buckets, edges = np.histogram(counts_list, bins=min(10, len(set(counts_list))))
    print("  Histogram (windows per stay):")
    for lo, hi, cnt in zip(edges[:-1], edges[1:], buckets):
        bar = "#" * min(int(cnt), 40)
        print(f"    [{int(lo):4d}-{int(hi):4d}]: {cnt:4d} stays  {bar}")

    feat_cols = [
        c for c in out.columns
        if any(c.endswith(f"_{s}") for s in ("mean", "stddev", "min", "max", "last", "slope"))
    ]
    has_any = None
    for c in feat_cols:
        cond = F.col(c).isNotNull()
        has_any = cond if has_any is None else (has_any | cond)
    if has_any is not None:
        n_covered = out.filter(has_any).count()
        print(
            f"  Rows with >=1 non-null feature: "
            f"{n_covered:,} / {total:,} ({n_covered/total*100:.1f}%)"
        )
    print("=" * 62)


# ── 12. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sepsis Structured Streaming pipeline")
    parser.add_argument(
        "--source", choices=["kafka", "csv"], default="kafka",
        help="Stream source: 'kafka' (default) or 'csv' (original file-source fallback)",
    )
    args = parser.parse_args()
    source = args.source

    print(f"Source mode: {source.upper()}")

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not COHORT_PATH.exists():
        sys.exit(f"ERROR: cohort.parquet not found at {COHORT_PATH}.\nRun step 01 first.")

    if source == "csv":
        csv_files = list(STREAM_INPUT.glob("*.csv"))
        if not csv_files:
            sys.exit(f"ERROR: no CSV files in {STREAM_INPUT}")
        print(f"Input: {len(csv_files)} CSV files in {STREAM_INPUT}")

    # ── Checkpoint directory (separate per source to avoid schema conflicts) ──
    CHECKPOINT = STREAM_OUTPUT / f"_checkpoint_{source}"

    # ── SparkSession ──────────────────────────────────────────────────────────
    spark = _build_spark(source)
    ui_url = getattr(spark.sparkContext, "uiWebUrl", "http://localhost:4040")
    print(f"Spark UI: {ui_url}")
    spark.streams.addListener(_ProgressListener())

    # ── Read stream ───────────────────────────────────────────────────────────
    if source == "kafka":
        raw = _build_kafka_stream(spark)
    else:
        raw = _build_csv_stream(spark)

    # ── Watermark (required for append output mode with windowed aggregation) ─
    raw = raw.withWatermark("charttime", WATERMARK_DELAY)

    # ── Window aggregation ────────────────────────────────────────────────────
    agg_exprs = _window_agg_exprs()
    if source == "kafka":
        # Extra latency stats per window (dropped before Parquet write to keep
        # schema identical to the CSV path)
        agg_exprs += [
            F.mean("latency_ms").alias("avg_latency_ms"),
            F.min("latency_ms") .alias("min_latency_ms"),
            F.max("latency_ms") .alias("max_latency_ms"),
            F.count("*")        .alias("n_events"),
        ]

    windowed = (
        raw
        .groupBy(
            "icustay_id",
            F.window(F.col("charttime"), WINDOW_DURATION, SLIDE_DURATION),
        )
        .agg(*agg_exprs)
        .withColumn("window_start_time", F.col("window.start"))
        .withColumn("window_end_time",   F.col("window.end"))
        .drop("window")
    )

    # ── Optional forward-fill ─────────────────────────────────────────────────
    if FORWARD_FILL_ENABLED:
        try:
            from pyspark.sql.streaming.state import GroupStateTimeout
        except ImportError:
            from pyspark.sql.streaming import GroupStateTimeout

        windowed = (
            windowed
            .withWatermark("window_end_time", "24 hours")
            .groupBy("icustay_id")
            .applyInPandasWithState(
                _forward_fill_fn,
                windowed.schema,
                _ffill_state_schema(),
                "append",
                GroupStateTimeout.EventTimeTimeout,
            )
        )

    # ── Label join ────────────────────────────────────────────────────────────
    labeled = _attach_labels(windowed, spark)

    # ── Sink ──────────────────────────────────────────────────────────────────
    STREAM_OUTPUT.mkdir(parents=True, exist_ok=True)

    if source == "kafka":
        # foreachBatch: write Parquet + latency CSV side-by-side
        query = (
            labeled.writeStream
            .foreachBatch(_make_foreachbatch(CHECKPOINT))
            .outputMode("append")
            .option("checkpointLocation", CHECKPOINT.as_posix())
            .trigger(availableNow=True)
            .start()
        )
    else:
        # CSV path: original Parquet sink (unchanged)
        query = (
            labeled.writeStream
            .format("parquet")
            .outputMode("append")
            .option("path",               STREAM_OUTPUT.as_posix())
            .option("checkpointLocation", CHECKPOINT.as_posix())
            .trigger(availableNow=True)
            .start()
        )

    print(f"Query running  (id={query.id})")
    query.awaitTermination()
    print("Query finished.")

    _print_report(spark)
    spark.stop()


if __name__ == "__main__":
    main()
