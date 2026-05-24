"""
03c_measure_latency.py — Compute streaming latency and throughput metrics.

Reads:
  data/processed/latency_log.csv     — per-window latency stats written by
                                        03_spark_streaming_job.py (Kafka path)
  data/processed/producer_timing.json — wall-clock timings from 03a_kafka_producer.py

Computes:
  1. Per-event latency percentiles: p50, p95, p99 (ms)
     (using per-window avg_latency_ms as proxy for per-event latency)
  2. End-to-end wall-clock: producer_start → last Parquet file write (seconds)
  3. Throughput: total_events / wall_clock_seconds (events/sec)

Saves:
  data/processed/latency_results.json

Usage:
  python src/03c_measure_latency.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED     = PROJECT_ROOT / "data" / "processed"
STREAM_OUTPUT = PROJECT_ROOT / "data" / "stream_output"

LATENCY_LOG_PATH    = PROCESSED / "latency_log.csv"
PRODUCER_TIMING_PATH = PROCESSED / "producer_timing.json"
RESULTS_PATH        = PROCESSED / "latency_results.json"


def _last_parquet_mtime_ms() -> int | None:
    parquet_files = list(STREAM_OUTPUT.glob("**/*.parquet"))
    if not parquet_files:
        return None
    return int(max(f.stat().st_mtime for f in parquet_files) * 1000)


def main() -> None:
    # ── Load latency log ──────────────────────────────────────────────────────
    if not LATENCY_LOG_PATH.exists():
        sys.exit(
            f"ERROR: latency_log.csv not found at {LATENCY_LOG_PATH}.\n"
            "Run 03_spark_streaming_job.py with --source kafka first."
        )
    lat_df = pd.read_csv(LATENCY_LOG_PATH)
    print(f"Latency log: {len(lat_df):,} window records")

    if "avg_latency_ms" not in lat_df.columns:
        sys.exit("ERROR: avg_latency_ms column missing from latency_log.csv")

    avg_latencies = lat_df["avg_latency_ms"].dropna().values
    if len(avg_latencies) == 0:
        sys.exit("ERROR: all avg_latency_ms values are null — check the Kafka run")

    p50 = float(np.percentile(avg_latencies, 50))
    p95 = float(np.percentile(avg_latencies, 95))
    p99 = float(np.percentile(avg_latencies, 99))
    p_mean = float(avg_latencies.mean())
    p_min  = float(avg_latencies.min())
    p_max  = float(avg_latencies.max())

    print(f"  avg_latency_ms  mean={p_mean:.1f}  "
          f"p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}  "
          f"min={p_min:.1f}  max={p_max:.1f}")

    # ── Load producer timing ──────────────────────────────────────────────────
    if not PRODUCER_TIMING_PATH.exists():
        sys.exit(
            f"ERROR: producer_timing.json not found at {PRODUCER_TIMING_PATH}.\n"
            "Run 03a_kafka_producer.py first."
        )
    with open(PRODUCER_TIMING_PATH) as f:
        prod = json.load(f)

    producer_start_ms = prod["producer_start_ms"]
    total_events      = prod["total_messages"]
    print(f"  Producer: {total_events:,} events  "
          f"producer_elapsed={prod['elapsed_seconds']:.1f}s  "
          f"avg_rate={prod['avg_throughput_msg_per_sec']:.0f} msg/s")

    # ── End-to-end wall-clock ─────────────────────────────────────────────────
    last_write_ms = _last_parquet_mtime_ms()
    if last_write_ms is None:
        print("  WARNING: no Parquet files found in stream_output; "
              "using producer_end_ms for wall-clock")
        last_write_ms = prod["producer_end_ms"]

    wall_clock_s = (last_write_ms - producer_start_ms) / 1000.0
    throughput   = total_events / wall_clock_s if wall_clock_s > 0 else 0.0

    print(f"  End-to-end wall-clock : {wall_clock_s:.1f}s")
    print(f"  Throughput            : {throughput:.0f} events/s")

    # ── Throughput per window (from latency log) ──────────────────────────────
    if "n_events" in lat_df.columns:
        window_throughputs = lat_df["n_events"].dropna()
        print(f"  Events per window     : mean={window_throughputs.mean():.1f}  "
              f"max={window_throughputs.max():.0f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "p50_latency_ms":             round(p50,  2),
        "p95_latency_ms":             round(p95,  2),
        "p99_latency_ms":             round(p99,  2),
        "mean_latency_ms":            round(p_mean, 2),
        "min_latency_ms":             round(p_min,  2),
        "max_latency_ms":             round(p_max,  2),
        "end_to_end_wall_clock_seconds": round(wall_clock_s, 1),
        "total_events":               total_events,
        "throughput_events_per_sec":  round(throughput, 1),
        "producer_start_ms":          producer_start_ms,
        "last_parquet_write_ms":      last_write_ms,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {RESULTS_PATH}")
    print("\n" + "=" * 50)
    print("LATENCY SUMMARY (for paper)")
    print("=" * 50)
    print(f"  End-to-end wall-clock : {wall_clock_s:.1f}s for {total_events:,} events")
    print(f"  Throughput            : {throughput:.0f} events/sec")
    print(f"  Per-event latency     : p50={p50:.0f} ms  p95={p95:.0f} ms  p99={p99:.0f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
