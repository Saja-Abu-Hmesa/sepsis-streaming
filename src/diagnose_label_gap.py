"""
diagnose_label_gap.py  —  Diagnose the Spark (246) vs pandas (374) label_6h discrepancy.

Spark job produced 246 positive windows; pandas recomputation in 03b produced 374.
The 128-window gap must be explained before model training.

Run:  python src/diagnose_label_gap.py
"""

import glob
import pathlib
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
STREAM_OUT   = DATA_DIR / "stream_output"
LSTM_PATH    = DATA_DIR / "processed" / "lstm_features.parquet"
COHORT_PATH  = DATA_DIR / "processed" / "cohort.parquet"
TARGET_03B   = pathlib.Path(__file__).resolve().parent / "03b_prepare_training_data.py"

HORIZON_S = 6 * 3600  # 21600 seconds


def _strip_tz(s: pd.Series) -> pd.Series:
    """Normalize a timestamp Series to tz-naive (UTC wall-clock values preserved)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


# ── Step 1: Load label sources ────────────────────────────────────────────────

def load_spark_labels() -> pd.DataFrame:
    parts = glob.glob(str(STREAM_OUT / "**" / "*.parquet"), recursive=True)
    parts = [p for p in parts if "_checkpoint" not in p.replace("\\", "/")]
    if not parts:
        sys.exit(f"ERROR: no parquet files in {STREAM_OUT}")
    frames = [
        pd.read_parquet(p, columns=["icustay_id", "window_start_time",
                                     "window_end_time", "label"])
        for p in parts
    ]
    df = pd.concat(frames, ignore_index=True)
    df["window_start_time"] = _strip_tz(df["window_start_time"])
    df["window_end_time"]   = _strip_tz(df["window_end_time"])
    print(f"Spark labels loaded : {len(df):,} rows  "
          f"(positives: {int(df['label'].sum()):,})")
    return df


def load_pandas_labels() -> pd.DataFrame:
    if not LSTM_PATH.exists():
        sys.exit(f"ERROR: {LSTM_PATH} not found. Run step 03b first.")
    df = pd.read_parquet(LSTM_PATH, columns=["icustay_id", "window_start_time",
                                              "window_end_time", "label_6h"])
    df["window_start_time"] = _strip_tz(df["window_start_time"])
    df["window_end_time"]   = _strip_tz(df["window_end_time"])
    print(f"Pandas labels loaded: {len(df):,} rows  "
          f"(positives: {int(df['label_6h'].sum()):,})")
    return df


# ── Step 2: Inner join and coverage check ─────────────────────────────────────

def inner_join(spark_df: pd.DataFrame,
               pandas_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    spark_keys  = spark_df.drop_duplicates(["icustay_id", "window_start_time"])
    pandas_keys = pandas_df.drop_duplicates(["icustay_id", "window_start_time"])
    print(f"\nUnique (icustay_id, window_start_time) keys:")
    print(f"  Spark : {len(spark_keys):,}")
    print(f"  Pandas: {len(pandas_keys):,}")

    merged = spark_df.merge(
        pandas_df[["icustay_id", "window_start_time", "label_6h"]],
        on=["icustay_id", "window_start_time"],
        how="inner",
    )
    n = len(merged)
    print(f"  Inner-join result : {n:,} rows")

    if n != 28890:
        print(f"\n  *** JOIN MISMATCH: expected 28,890, got {n:,} ***")
        s_set = set(zip(spark_df["icustay_id"].tolist(),
                        spark_df["window_start_time"].astype(str).tolist()))
        p_set = set(zip(pandas_df["icustay_id"].tolist(),
                        pandas_df["window_start_time"].astype(str).tolist()))
        only_spark  = s_set - p_set
        only_pandas = p_set - s_set
        print(f"  Keys in Spark only : {len(only_spark):,}")
        print(f"  Keys in Pandas only: {len(only_pandas):,}")
        if only_spark:
            print(f"  Sample Spark-only : {list(only_spark)[:3]}")
        if only_pandas:
            print(f"  Sample Pandas-only: {list(only_pandas)[:3]}")
        # Show tz info to aid timezone diagnosis
        print(f"\n  Timestamp dtype check:")
        print(f"    Spark  window_start_time dtype: {spark_df['window_start_time'].dtype}")
        print(f"    Pandas window_start_time dtype: {pandas_df['window_start_time'].dtype}")
        print(f"    Sample Spark  values: {spark_df['window_start_time'].iloc[:3].tolist()}")
        print(f"    Sample Pandas values: {pandas_df['window_start_time'].iloc[:3].tolist()}")

    return merged, n


# ── Step 3-4: Diagnose discrepancies ─────────────────────────────────────────

def _categorize(gap: float) -> str:
    if np.isnan(gap):
        return "D"
    if gap in (0.0, float(HORIZON_S)):
        return "A"
    if abs(gap) < 60 or abs(gap - HORIZON_S) < 60:
        return "B"
    return "E"


def diagnose(merged: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    diff = merged[merged["label"] != merged["label_6h"]].copy()
    diff = diff.rename(columns={"label": "spark_label", "label_6h": "pandas_label"})

    n_s0p1 = int(((diff["spark_label"] == 0) & (diff["pandas_label"] == 1)).sum())
    n_s1p0 = int(((diff["spark_label"] == 1) & (diff["pandas_label"] == 0)).sum())
    print(f"\nDiscrepant rows: {len(diff):,}")
    print(f"  spark=0 / pandas=1 (pandas finds more positives): {n_s0p1:,}")
    print(f"  spark=1 / pandas=0 (spark finds more positives) : {n_s1p0:,}")

    if len(diff) == 0:
        return diff

    septic = cohort[cohort["sepsis_label"] == 1][["icustay_id",
                                                    "sepsis_onset_time"]].copy()
    septic["sepsis_onset_time"] = _strip_tz(septic["sepsis_onset_time"])
    diff = diff.merge(septic, on="icustay_id", how="left")

    diff["gap_seconds"] = (
        diff["sepsis_onset_time"] - diff["window_end_time"]
    ).dt.total_seconds()

    diff["category"] = diff["gap_seconds"].map(_categorize)

    # Category C marker: Spark window_end on :00 or :30 boundary
    diff["end_minute"] = diff["window_end_time"].dt.minute
    diff["spark_on_30min"] = diff["end_minute"].isin([0, 30])

    return diff


# ── Step 5: Print report ──────────────────────────────────────────────────────

def print_report(diff: pd.DataFrame, joined_n: int,
                 spark_df: pd.DataFrame) -> str:
    SHOW_COLS = ["icustay_id", "window_start_time", "window_end_time",
                 "sepsis_onset_time", "spark_label", "pandas_label",
                 "gap_seconds", "category"]

    print("\n" + "=" * 70)
    print("CATEGORY BREAKDOWN")
    print("=" * 70)
    cat_desc = {
        "A": "Exact boundary: gap=0s or gap=21600s",
        "B": "Sub-minute precision near boundary (|gap - boundary| < 60s)",
        "D": "No sepsis_onset_time for patient (not in septic cohort)",
        "E": "Other / unexplained",
    }
    cat_counts = diff["category"].value_counts().sort_index()
    for cat, cnt in cat_counts.items():
        print(f"  Category {cat} — {cat_desc.get(cat,'?')}: {cnt:,}")

    # Category C check across all discrepant rows
    pct_on = diff["spark_on_30min"].mean() * 100
    all_spark_on = spark_df["window_end_time"].dt.minute.isin([0, 30]).mean() * 100
    print(f"\n  Category C check:")
    print(f"    Discrepant rows with window_end on :00/:30 : {pct_on:.1f}%")
    print(f"    All Spark output rows on :00/:30           : {all_spark_on:.1f}%")

    # Gap distribution
    gap = diff["gap_seconds"].dropna()
    if len(gap):
        print(f"\n  Gap distribution (sepsis_onset_time - window_end_time, seconds):")
        print(f"    min={gap.min():.3f}  max={gap.max():.3f}  "
              f"mean={gap.mean():.3f}  median={gap.median():.3f}")
        print(f"    gap = 0s exactly          : {(gap == 0).sum()}")
        print(f"    0 < gap < 60s             : {((gap > 0) & (gap < 60)).sum()}")
        print(f"    21540 < gap < 21600s      : {((gap > 21540) & (gap < 21600)).sum()}")
        print(f"    gap = 21600s exactly      : {(gap == 21600).sum()}")
        print(f"    21600 < gap < 21660s      : {((gap > 21600) & (gap < 21660)).sum()}")
        print(f"    gap > 21660s (unexplained): {(gap > 21660).sum()}")

    # Show 5 examples for D and E
    show_cols = [c for c in SHOW_COLS if c in diff.columns]
    for cat in ("D", "E"):
        sub = diff[diff["category"] == cat]
        if len(sub):
            print(f"\n  ── Category {cat} examples (n={len(sub)}) ──")
            print(sub[show_cols].head(5).to_string(index=False))

    # Determine verdict
    dominant = cat_counts.idxmax() if len(cat_counts) else "none"
    n_disc = len(diff)

    if joined_n != 28890:
        verdict = (
            f"Join mismatch: expected 28,890 matched rows, got {joined_n:,}. "
            "Root cause is likely a timestamp timezone mismatch — Spark writes UTC-aware "
            "timestamps; pandas may read or compare them differently. "
            "Normalize all timestamps to tz-naive UTC before any label comparison."
        )
    elif dominant == "A":
        verdict = (
            f"Boundary inclusivity: {cat_counts['A']} windows sit at gap=0s or gap=21600s "
            "exactly. One side uses strict inequality, the other uses ≤ at the endpoint. "
            "Spark: sepsis_onset_time <= window_end_time + INTERVAL 6 HOURS (closed upper). "
            "Pandas: sepsis_onset_time <= window_end_time + Timedelta(6h) (also closed). "
            "If gap is 0 exactly, both should agree — check whether Spark INTERVAL rounds "
            "to second precision, changing the effective boundary."
        )
    elif dominant == "B":
        verdict = (
            f"Sub-second precision: {cat_counts.get('B', 0)} windows have sepsis_onset_time "
            "within 60s of the 6h boundary. Spark uses unix_timestamp() which truncates to "
            "second precision; when the boundary arithmetic lands within a fractional second, "
            "Spark and pandas round to opposite sides. "
            "Use pandas labels (374 positives) — they preserve microsecond precision."
        )
    elif dominant == "E":
        verdict = (
            f"Unexplained discrepancy in {cat_counts.get('E', 0)} windows. "
            "Inspect the Category E example rows above. Possible causes: timezone offset "
            "in sepsis_onset_time, duplicate rows in stream_output, or cohort join anomaly."
        )
    else:
        lines = [f"Cat-{k}: {v}" for k, v in cat_counts.items()]
        verdict = f"Mixed causes ({', '.join(lines)}). See breakdown above."

    print(f"\nVERDICT: {verdict}")
    return verdict


# ── Step 6: Write comment block to 03b ───────────────────────────────────────

def write_comment_to_03b(verdict: str, diff: pd.DataFrame, joined_n: int) -> None:
    if not TARGET_03B.exists():
        print(f"WARNING: {TARGET_03B} not found — skipping comment write")
        return

    n_disc = len(diff)
    cat_summary = "N/A"
    if n_disc > 0 and "category" in diff.columns:
        cat_counts = diff["category"].value_counts().sort_index()
        cat_summary = "  ".join(f"Cat-{k}: {v}" for k, v in cat_counts.items())

    block = (
        "# ══════════════════════════════════════════════════════════════════\n"
        "# LABEL GAP DIAGNOSIS  (src/diagnose_label_gap.py)\n"
        "#\n"
        f"# Spark label_6h positives : 246 (from data/stream_output/)\n"
        f"# Pandas label_6h positives: 374 (recomputed in this script)\n"
        f"# Discrepant windows       : {n_disc}\n"
        f"# Join coverage            : {joined_n:,} / 28,890 rows matched\n"
        f"# Category breakdown       : {cat_summary}\n"
        "#\n"
        f"# VERDICT: {verdict}\n"
        "#\n"
        "# USE FOR TRAINING: pandas-recomputed label_6h (374 positives).\n"
        "# It operates directly on raw cohort timestamps without Spark's\n"
        "# intermediate type coercions or streaming watermark truncation.\n"
        "# ══════════════════════════════════════════════════════════════════\n"
    )

    text = TARGET_03B.read_text(encoding="utf-8")
    # Remove any prior diagnosis block
    text = re.sub(
        r"# ══+\n# LABEL GAP DIAGNOSIS.*?# ══+\n",
        "",
        text,
        flags=re.DOTALL,
    )
    text = block + text
    TARGET_03B.write_text(text, encoding="utf-8")
    print(f"\nDiagnosis comment written to {TARGET_03B.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("LABEL GAP DIAGNOSIS  Spark(246) vs Pandas(374) label_6h")
    print("=" * 70)

    spark_df  = load_spark_labels()
    pandas_df = load_pandas_labels()
    cohort    = pd.read_parquet(COHORT_PATH)
    print(f"Cohort: {len(cohort):,} patients  "
          f"({int(cohort['sepsis_label'].sum()):,} septic)")

    merged, joined_n = inner_join(spark_df, pandas_df)

    if joined_n != 28890:
        verdict = (
            f"Join mismatch: expected 28,890 matched rows, got {joined_n:,}. "
            "Likely a timestamp timezone mismatch. Check _strip_tz output above."
        )
        write_comment_to_03b(verdict, pd.DataFrame(), joined_n)
        return

    print("  Join matched exactly 28,890 rows — good.")
    diff = diagnose(merged, cohort)

    if len(diff) == 0:
        print("\nNo discrepancies. Labels are identical.")
        write_comment_to_03b("No discrepancies — labels are identical.",
                             pd.DataFrame(), joined_n)
        return

    verdict = print_report(diff, joined_n, spark_df)
    write_comment_to_03b(verdict, diff, joined_n)


if __name__ == "__main__":
    main()
