"""
03a_kafka_producer.py — Replay stream_input CSV chunks to Kafka topic via Spark.

Uses the same spark-sql-kafka connector as the consumer (no kafka-python needed).
Reads all 50 CSV chunks as a Spark batch DataFrame, stamps each row with
producer_timestamp_ms = current wall-clock ms, serialises to JSON, and writes
to topic 'mimic-icu-stream' in one batch write.

Requirements:
  - SPARK_HOME set, spark-sql-kafka-0-10_2.13:4.1.1 on classpath
    (auto-downloaded on first run if internet is available)
  - Kafka broker reachable at localhost:9092

Usage:
  python src/03a_kafka_producer.py
  python src/03a_kafka_producer.py --bootstrap localhost:9092 --rate 5000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Spark bootstrap (mirrors 03_spark_streaming_job.py) ──────────────────────
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
        sys.exit(f"ERROR: pyspark.zip not found at {_pyspark_zip}.")
    sys.path.insert(0, str(_pyspark_zip))
    if _py4j_zips:
        sys.path.insert(0, str(_py4j_zips[0]))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType, TimestampType

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
STREAM_INPUT  = PROJECT_ROOT / "data" / "stream_input"
PROCESSED     = PROJECT_ROOT / "data" / "processed"
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"
TOPIC         = "mimic-icu-stream"

CSV_SCHEMA = StructType([
    StructField("icustay_id",   LongType(),      True),
    StructField("charttime",    TimestampType(), True),
    StructField("signal_name",  StringType(),    True),
    StructField("valuenum",     DoubleType(),    True),
    StructField("source_table", StringType(),    True),
])


def main() -> None:
    parser = argparse.ArgumentParser(description="Kafka producer (Spark batch write)")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--rate", type=int, default=5_000,
                        help="Unused in Spark batch mode — kept for CLI compatibility")
    args = parser.parse_args()

    csv_files = sorted(STREAM_INPUT.glob("part-*.csv"))
    if not csv_files:
        sys.exit(f"ERROR: no part-*.csv files in {STREAM_INPUT}")
    print(f"Producer (Spark batch mode)")
    print(f"  Broker : {args.bootstrap}")
    print(f"  Topic  : {TOPIC}")
    print(f"  Chunks : {len(csv_files)} CSV files in {STREAM_INPUT}")

    # ── SparkSession ──────────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("sepsis-kafka-producer")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"  Spark UI: {getattr(spark.sparkContext, 'uiWebUrl', 'http://localhost:4040')}")

    # ── Read all CSV chunks as a batch DataFrame ───────────────────────────────
    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .option("pathGlobFilter", "part-*.csv")
        .schema(CSV_SCHEMA)
        .load(STREAM_INPUT.as_posix())
    )

    total_rows = df.count()
    print(f"  Rows loaded : {total_rows:,}")

    wall_start_ms = int(time.time() * 1000)

    # ── Build JSON payload column (matches KAFKA_JSON_SCHEMA in consumer) ─────
    # producer_timestamp_ms: wall-clock ms at the time Spark evaluates this row.
    # current_timestamp() is evaluated per-partition at execution time.
    df_with_ts = (
        df
        .withColumn("producer_timestamp_ms",
                    (F.unix_timestamp(F.current_timestamp()) * 1000).cast(LongType()))
        .withColumn("charttime_str",
                    F.date_format(F.col("charttime"), "yyyy-MM-dd HH:mm:ss"))
    )

    df_kafka = df_with_ts.select(
        F.to_json(
            F.struct(
                F.col("icustay_id"),
                F.col("charttime_str").alias("charttime"),
                F.col("signal_name"),
                F.col("valuenum"),
                F.col("source_table"),
                F.col("producer_timestamp_ms"),
            )
        ).alias("value")
    )

    # ── Write to Kafka ────────────────────────────────────────────────────────
    print(f"  Writing {total_rows:,} messages to Kafka …")
    t0 = time.time()
    (
        df_kafka.write
        .format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("topic", TOPIC)
        .save()
    )
    elapsed = time.time() - t0
    wall_end_ms = int(time.time() * 1000)
    avg_rate = total_rows / elapsed if elapsed > 0 else 0

    print(f"\nProducer complete.")
    print(f"  Total messages : {total_rows:,}")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print(f"  Avg throughput : {avg_rate:.0f} msg/s")
    print(f"  Start (ms)     : {wall_start_ms}")
    print(f"  End   (ms)     : {wall_end_ms}")

    # ── Save timing for 03c_measure_latency.py ───────────────────────────────
    PROCESSED.mkdir(parents=True, exist_ok=True)
    timing_path = PROCESSED / "producer_timing.json"
    with open(timing_path, "w") as f:
        json.dump({
            "producer_start_ms":          wall_start_ms,
            "producer_end_ms":            wall_end_ms,
            "total_messages":             total_rows,
            "elapsed_seconds":            round(elapsed, 3),
            "avg_throughput_msg_per_sec": round(avg_rate, 1),
        }, f, indent=2)
    print(f"  Timing saved   : {timing_path}")

    spark.stop()


if __name__ == "__main__":
    main()
