"""
07_realtime_inference.py  —  Real-time sepsis risk scoring via Spark + XGBoost.

Pipeline:
  Kafka topic mimic-icu-stream (raw vitals)
    → readStream("kafka") + JSON parse
    → watermark + sliding-window aggregation  (identical to step 03)
    → foreachBatch:
        • add temporal features  (window_idx, window_idx_norm, hours_in_icu)
        • XGBoost predict_proba  (label_6h / label_12h / label_24h models)
        • print high-risk alerts to console
        • append predictions to data/processed/realtime_predictions.csv

Usage (from project root, after starting Kafka and running the producer):
  # Terminal 1 – start Kafka (if not already running)
  & "e:\kafka\bin\windows\kafka-server-start.bat" "e:\kafka\config\kraft\server.properties"

  # Terminal 2 – produce vitals
  .\.venv\Scripts\python src/03a_kafka_producer.py

  # Terminal 3 – real-time inference
  $env:SPARK_HOME = "E:\Big Data\spark"
  .\.venv\Scripts\python src/07_realtime_inference.py

  # Optional flags:
  #   --threshold 0.30     risk score threshold for alerts (default 0.15)
  #   --bootstrap localhost:9092
  #   --horizon 6h         which label horizon to score (6h / 12h / 24h, default 6h)
"""

import argparse
import os
import sys
from pathlib import Path

# ── Spark bootstrap ───────────────────────────────────────────────────────────
SPARK_HOME = os.environ.get("SPARK_HOME")
if not SPARK_HOME:
    sys.exit(
        "ERROR: SPARK_HOME is not set.\n"
        "  PowerShell: $env:SPARK_HOME = 'E:\\Big Data\\spark'"
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
        sys.exit(f"ERROR: pyspark.zip not found at {_pyspark_zip}")
    sys.path.insert(0, str(_pyspark_zip))
    if _py4j_zips:
        sys.path.insert(0, str(_py4j_zips[0]))

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, LongType, StringType,
    StructField, StructType, TimestampType,
)

try:
    import xgboost as xgb
except ImportError:
    sys.exit("ERROR: xgboost not installed. Run: pip install xgboost")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED     = DATA_DIR / "processed"
MODEL_DIR     = PROCESSED / "models"
STREAM_OUTPUT = DATA_DIR / "stream_output"
PREDICTIONS   = PROCESSED / "realtime_predictions.csv"

KAFKA_BOOTSTRAP  = "localhost:9092"
KAFKA_TOPIC      = "mimic-icu-stream"
KAFKA_PACKAGE    = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"

SIGNALS: list[str] = [
    "bilirubin", "creatinine", "dbp", "heart_rate",
    "lactate", "map", "platelets", "resp_rate",
    "sbp", "spo2", "temperature", "wbc",
]

# XGBoost was trained on mean/min/max/last per signal (no stddev/slope)
# plus 3 temporal features added by add_temporal_features().
XGB_FEATURE_COLS: list[str] = (
    [f"{s}_{stat}" for s in SIGNALS for stat in ("mean", "min", "max", "last")]
    + ["window_idx", "window_idx_norm", "hours_in_icu"]
)

WINDOW_DURATION = "1 hour"
SLIDE_DURATION  = "30 minutes"
WATERMARK_DELAY = "1 hour"

ALERT_EMOJI = "🚨"   # shown in console only

# ── Kafka JSON schema (must match 03a_kafka_producer.py) ─────────────────────
KAFKA_JSON_SCHEMA = StructType([
    StructField("icustay_id",            LongType(),   True),
    StructField("charttime",             StringType(), True),
    StructField("signal_name",           StringType(), True),
    StructField("valuenum",              DoubleType(), True),
    StructField("source_table",          StringType(), True),
    StructField("producer_timestamp_ms", LongType(),   True),
])


# ── Window aggregation (identical to 03_spark_streaming_job.py) ──────────────
def _window_agg_exprs() -> list:
    exprs = []
    for sig in SIGNALS:
        val    = F.when(F.col("signal_name") == sig, F.col("valuenum"))
        t_unix = F.when(
            F.col("signal_name") == sig,
            F.unix_timestamp(F.col("charttime")).cast(DoubleType()),
        )
        exprs += [
            F.mean(val)                        .alias(f"{sig}_mean"),
            F.stddev(val)                      .alias(f"{sig}_stddev"),
            F.min(val)                         .alias(f"{sig}_min"),
            F.max(val)                         .alias(f"{sig}_max"),
            F.last(val, ignorenulls=True)      .alias(f"{sig}_last"),
            (F.covar_samp(t_unix, val) / F.var_samp(t_unix)).alias(f"{sig}_slope"),
        ]
    # latency tracking
    exprs += [
        F.mean("latency_ms").alias("avg_latency_ms"),
        F.count("*")        .alias("n_events"),
    ]
    return exprs


# ── Temporal features (mirrors 04_train_xgboost.py: add_temporal_features) ───
def _add_temporal_features(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values(["icustay_id", "window_start_time"]).copy()
    pdf["window_idx"] = pdf.groupby("icustay_id").cumcount()
    stay_len = pdf.groupby("icustay_id")["window_idx"].transform("max").clip(lower=1)
    pdf["window_idx_norm"] = pdf["window_idx"] / stay_len
    first_ts = pdf.groupby("icustay_id")["window_start_time"].transform("min")
    pdf["hours_in_icu"] = (
        (pdf["window_start_time"] - first_ts).dt.total_seconds() / 3600.0
    )
    return pdf


# ── foreachBatch inference callback ──────────────────────────────────────────
def _make_inference_fn(models: dict[str, xgb.XGBClassifier],
                       feature_names: list[str],
                       horizon: str,
                       threshold: float,
                       first_run: list):
    """
    Returns a foreachBatch function.
    models        — {"6h": model, "12h": model, "24h": model}
    feature_names — exact columns the models were trained on
    horizon       — which model to use for alerts ("6h" / "12h" / "24h")
    threshold     — risk score above which an alert is printed
    first_run     — mutable list used to detect first call (write CSV header once)
    """
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)

    def _infer(df, epoch_id: int) -> None:
        pdf = df.toPandas()
        if pdf.empty:
            return

        # ensure window_start_time is datetime
        for col in ("window_start_time", "window_end_time"):
            if col in pdf.columns and not pd.api.types.is_datetime64_any_dtype(pdf[col]):
                pdf[col] = pd.to_datetime(pdf[col])

        pdf = _add_temporal_features(pdf)

        # score with all three horizon models
        for h, model in models.items():
            missing = [c for c in feature_names if c not in pdf.columns]
            for c in missing:
                pdf[c] = np.nan
            X = pdf[feature_names].astype(float)
            pdf[f"score_{h}"] = model.predict_proba(X)[:, 1]

        # alert column for the chosen horizon
        score_col = f"score_{horizon}"
        alert_mask = pdf[score_col] >= threshold
        n_alerts   = alert_mask.sum()

        print(f"\n[Batch {epoch_id}]  windows={len(pdf):,}  "
              f"alerts({horizon} ≥{threshold:.2f})={n_alerts}", flush=True)

        if n_alerts:
            alerts = pdf[alert_mask].sort_values(score_col, ascending=False)
            for _, row in alerts.head(10).iterrows():
                print(
                    f"  {ALERT_EMOJI}  ICU stay {int(row['icustay_id'])}  "
                    f"window_end={row['window_end_time']}  "
                    f"score_6h={row.get('score_6h', float('nan')):.3f}  "
                    f"score_12h={row.get('score_12h', float('nan')):.3f}  "
                    f"score_24h={row.get('score_24h', float('nan')):.3f}",
                    flush=True,
                )

        # persist all predictions
        out_cols = [
            "icustay_id", "window_start_time", "window_end_time",
            "n_events", "avg_latency_ms",
            "score_6h", "score_12h", "score_24h",
        ]
        out = pdf[[c for c in out_cols if c in pdf.columns]].copy()
        out["alert"] = alert_mask.astype(int)
        write_header = bool(first_run)
        if first_run:
            if PREDICTIONS.exists():
                PREDICTIONS.unlink()
            first_run.clear()
        out.to_csv(str(PREDICTIONS), mode="a", header=write_header, index=False)

    return _infer


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time sepsis inference")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Risk score alert threshold (default 0.30 = ~90%% specificity)")
    parser.add_argument("--horizon", choices=["6h", "12h", "24h"], default="6h",
                        help="Label horizon model used for alerts")
    args = parser.parse_args()

    # ── Load XGBoost models ───────────────────────────────────────────────────
    models: dict[str, xgb.XGBClassifier] = {}
    for h in ("6h", "12h", "24h"):
        path = MODEL_DIR / f"xgb_label_{h}.json"
        if not path.exists():
            sys.exit(f"ERROR: model not found: {path}\nRun step 04 first.")
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        models[h] = m
        print(f"Loaded model: {path.name}")

    feature_names: list[str] = XGB_FEATURE_COLS
    print(f"Feature set: {len(feature_names)} cols  "
          f"(48 signal stats + 3 temporal)  first={feature_names[:3]}")

    # ── SparkSession ──────────────────────────────────────────────────────────
    CHECKPOINT = STREAM_OUTPUT / "_checkpoint_inference"
    STREAM_OUTPUT.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("sepsis-realtime-inference")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark UI : {getattr(spark.sparkContext, 'uiWebUrl', 'http://localhost:4040')}")
    print(f"Broker   : {args.bootstrap}")
    print(f"Topic    : {KAFKA_TOPIC}")
    print(f"Threshold: {args.threshold}  Horizon: {args.horizon}")
    print(f"Predictions → {PREDICTIONS}\n")

    # ── Kafka source ──────────────────────────────────────────────────────────
    raw_kafka = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
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
        .withWatermark("charttime", WATERMARK_DELAY)
    )

    # ── Windowed aggregation ──────────────────────────────────────────────────
    windowed = (
        parsed
        .groupBy(
            "icustay_id",
            F.window(F.col("charttime"), WINDOW_DURATION, SLIDE_DURATION),
        )
        .agg(*_window_agg_exprs())
        .withColumn("window_start_time", F.col("window.start"))
        .withColumn("window_end_time",   F.col("window.end"))
        .drop("window")
    )

    # ── Streaming sink with inference ─────────────────────────────────────────
    infer_fn = _make_inference_fn(
        models, feature_names, args.horizon, args.threshold, [True]
    )

    query = (
        windowed.writeStream
        .foreachBatch(infer_fn)
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT.as_posix())
        .trigger(availableNow=True)
        .start()
    )

    print(f"Query started (id={query.id})  — waiting for data …\n")
    query.awaitTermination()
    print("\nInference complete.")
    print(f"Predictions saved to: {PREDICTIONS}")

    if PREDICTIONS.exists():
        df = pd.read_csv(PREDICTIONS)
        print(f"\nSummary: {len(df):,} windows scored")
        for h in ("6h", "12h", "24h"):
            col = f"score_{h}"
            if col in df.columns:
                alerts = (df[col] >= args.threshold).sum()
                print(f"  score_{h}:  mean={df[col].mean():.4f}  "
                      f"max={df[col].max():.4f}  alerts={alerts:,}")

    spark.stop()


if __name__ == "__main__":
    main()
