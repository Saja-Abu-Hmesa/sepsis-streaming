# ══════════════════════════════════════════════════════════════════
# LABEL GAP DIAGNOSIS  (src/diagnose_label_gap.py)
#
# Spark label_6h positives : 246 (from data/stream_output/)
# Pandas label_6h positives: 374 (recomputed in this script)
# Discrepant windows       : 336
# Join coverage            : 28,890 / 28,890 rows matched
# Category breakdown       : Cat-E: 336
#
# VERDICT: Unexplained discrepancy in 336 windows. Inspect the Category E example rows above. Possible causes: timezone offset in sepsis_onset_time, duplicate rows in stream_output, or cohort join anomaly.
#
# USE FOR TRAINING: pandas-recomputed label_6h (374 positives).
# It operates directly on raw cohort timestamps without Spark's
# intermediate type coercions or streaming watermark truncation.
# ══════════════════════════════════════════════════════════════════
"""
03b_prepare_training_data.py  —  Model-side preprocessing for sepsis LSTM and XGBoost.

This is the preprocessing step referenced in paper Section III.B, framing
forward-fill as a model-side operation rather than a change to the Spark pipeline
(consistent with the SEPRES architecture).

Forward-fill with binary missingness indicators follows:
  Che et al. 2018. "Recurrent Neural Networks for Multivariate Time Series
  with Missing Values." Scientific Reports 8, 6085.
  DOI: 10.1038/s41598-018-24271-9

Multi-horizon labels follow:
  Dalal et al. (cited as [19] in paper): 6h / 12h / 24h sepsis prediction.

Outputs:
  data/processed/lstm_features.parquet   — 12 _last + 12 _fresh + labels
  data/processed/xgb_features.parquet    — 48 raw features (mean/min/max/last) + labels
  data/processed/folds.parquet           — icustay_id, fold (0-4)

Usage:
  python src/03b_prepare_training_data.py
"""

import glob
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
STREAM_OUT    = DATA_DIR / "stream_output"
COHORT_PATH   = DATA_DIR / "processed" / "cohort.parquet"
OUT_DIR       = DATA_DIR / "processed"

SIGNALS: list[str] = [
    "bilirubin", "creatinine", "dbp", "heart_rate",
    "lactate", "map", "platelets", "resp_rate",
    "sbp", "spo2", "temperature", "wbc",
]

LABEL_HORIZONS = {
    "label_6h":  pd.Timedelta(hours=6),
    "label_12h": pd.Timedelta(hours=12),
    "label_24h": pd.Timedelta(hours=24),
}

N_FOLDS = 5
RANDOM_STATE = 42


# ── Step 1: Load & sort ───────────────────────────────────────────────────────

def load_stream_output() -> pd.DataFrame:
    parts = glob.glob(str(STREAM_OUT / "**" / "*.parquet"), recursive=True)
    if not parts:
        sys.exit(f"ERROR: no parquet files in {STREAM_OUT}. Run step 03 first.")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.sort_values(["icustay_id", "window_start_time"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} windows from {len(parts)} parquet files.")
    return df


# ── Step 2: Drop high-null, low-signal columns ────────────────────────────────

def drop_noisy_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns
                 if c.endswith("_stddev") or c.endswith("_slope")]
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} columns (*_stddev, *_slope). "
          f"Remaining: {df.shape[1]}")
    return df


# ── Step 3: Multi-horizon labels ─────────────────────────────────────────────

def attach_multi_horizon_labels(df: pd.DataFrame,
                                cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Derive label_6h / label_12h / label_24h from cohort.sepsis_onset_time.
    Each label = 1 iff sepsis_label==1 AND
                 window_end_time < sepsis_onset_time <= window_end_time + horizon.
    Supersedes the single `label` column from the Spark job.
    """
    meta = cohort[["icustay_id", "sepsis_label", "sepsis_onset_time"]].copy()
    df = df.drop(columns=["label"], errors="ignore")
    df = df.merge(meta, on="icustay_id", how="left")

    for col, horizon in LABEL_HORIZONS.items():
        df[col] = (
            (df["sepsis_label"] == 1)
            & (df["sepsis_onset_time"] > df["window_end_time"])
            & (df["sepsis_onset_time"] <= df["window_end_time"] + horizon)
        ).astype(np.int8)

    df = df.drop(columns=["sepsis_label", "sepsis_onset_time"])
    for name, horizon in LABEL_HORIZONS.items():
        n = int(df[name].sum())
        print(f"  {name}: {n:,} positive windows ({n/len(df)*100:.2f}%)")
    return df


# ── Step 4: Patient-level stratified folds ────────────────────────────────────

def build_folds(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame: icustay_id, fold (0–4), stratified on sepsis_label."""
    pts = cohort[["icustay_id", "sepsis_label"]].drop_duplicates("icustay_id")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    X = pts["icustay_id"].values
    y = pts["sepsis_label"].values
    fold_col = np.empty(len(X), dtype=np.int8)
    for fold_id, (_, test_idx) in enumerate(skf.split(X, y)):
        fold_col[test_idx] = fold_id
    pts = pts.copy()
    pts["fold"] = fold_col
    return pts[["icustay_id", "fold"]]


# ── Step 5: Cross-fold median imputation (leakage-free) ──────────────────────

def _cross_fold_medians(df_orig: pd.DataFrame,
                        fold_df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """
    For each fold k, compute the population median of each signal's _last column
    from the patients in the OTHER four folds (train split).
    df_orig must be the pre-forward-fill snapshot to avoid contamination.
    Returns: {fold_id: {signal: median_float}}
    """
    medians: dict[int, dict[str, float]] = {}
    for fold_id in range(N_FOLDS):
        train_ids = set(
            fold_df.loc[fold_df["fold"] != fold_id, "icustay_id"]
        )
        train_df = df_orig[df_orig["icustay_id"].isin(train_ids)]
        fold_meds: dict[str, float] = {}
        for sig in SIGNALS:
            col = f"{sig}_last"
            med = float(train_df[col].median())   # NaN → 0 fallback below
            fold_meds[sig] = med if np.isfinite(med) else 0.0
        medians[fold_id] = fold_meds
    return medians


def _verify_leakage(df_orig: pd.DataFrame,
                    fold_df: pd.DataFrame,
                    medians: dict[int, dict[str, float]]) -> None:
    """Assert that stored medians match independently-recomputed cross-fold medians."""
    for fold_id in range(N_FOLDS):
        train_ids = set(
            fold_df.loc[fold_df["fold"] != fold_id, "icustay_id"]
        )
        train_df = df_orig[df_orig["icustay_id"].isin(train_ids)]
        for sig in SIGNALS:
            col = f"{sig}_last"
            expected = train_df[col].median()
            stored   = medians[fold_id][sig]
            # Handle both-NaN case (signal fully absent from train)
            if np.isnan(expected):
                expected = 0.0
            if abs(float(expected) - stored) > 1e-9:
                raise AssertionError(
                    f"Leakage detected: fold={fold_id} signal={sig} "
                    f"stored_median={stored:.6f} != recomputed={expected:.6f}"
                )
    print("Leakage check passed: all imputation medians verified against "
          "independently-recomputed cross-fold training medians.")


def forward_fill_with_imputation(df: pd.DataFrame,
                                 df_orig: pd.DataFrame,
                                 fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient forward-fill of _last columns, then cross-fold median imputation
    for any leading nulls (windows before the first observation of that signal).

    Following Che et al. 2018: forward-fill propagates the last observed value;
    leading nulls before first observation are filled with the train-population
    median rather than zero to avoid systematic bias in the LSTM embedding.

    Args:
        df:      working copy of the windowed dataset (will be mutated)
        df_orig: snapshot of df BEFORE any ffill (used to compute train medians)
        fold_df: icustay_id → fold assignment
    """
    medians = _cross_fold_medians(df_orig, fold_df)
    _verify_leakage(df_orig, fold_df, medians)

    # Build lookup: icustay_id → fold_id
    id_to_fold = fold_df.set_index("icustay_id")["fold"].to_dict()

    for sig in SIGNALS:
        col = f"{sig}_last"
        # Forward-fill within each patient's time series
        df[col] = df.groupby("icustay_id", sort=False)[col].ffill()

    # Impute remaining leading nulls with cross-fold train median
    for fold_id in range(N_FOLDS):
        test_ids = set(fold_df.loc[fold_df["fold"] == fold_id, "icustay_id"])
        fold_mask = df["icustay_id"].isin(test_ids)
        for sig in SIGNALS:
            col = f"{sig}_last"
            null_mask = fold_mask & df[col].isna()
            if null_mask.any():
                df.loc[null_mask, col] = medians[fold_id][sig]

    return df


# ── Step 6: LSTM feature set ──────────────────────────────────────────────────

def build_lstm_features(df_raw: pd.DataFrame,
                        fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    12 forward-filled _last columns + 12 binary _fresh indicators.
    _fresh[t] = 1 iff the signal was actually observed in this window
    (i.e., _last was non-null before forward-fill), 0 otherwise.
    Leading nulls → train-fold population median (Che et al. 2018).
    """
    id_cols    = ["icustay_id", "window_start_time", "window_end_time"]
    label_cols = list(LABEL_HORIZONS.keys())
    last_cols  = [f"{s}_last" for s in SIGNALS]

    df = df_raw[id_cols + label_cols + last_cols].copy()

    # Compute _fresh BEFORE forward-fill: 1 iff signal observed in this window
    for sig in SIGNALS:
        col = f"{sig}_last"
        df[f"{sig}_fresh"] = df[col].notna().astype(np.int8)

    # Snapshot pre-ffill for median computation
    df_orig = df[id_cols + last_cols].copy()

    # Forward-fill + cross-fold median imputation
    df = forward_fill_with_imputation(df, df_orig, fold_df)

    # Column order: IDs → labels → features
    fresh_cols = [f"{s}_fresh" for s in SIGNALS]
    ordered    = id_cols + label_cols + last_cols + fresh_cols
    return df[ordered]


# ── Step 7: XGBoost feature set ───────────────────────────────────────────────

def build_xgb_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    48 raw window features (mean / min / max / last) — NO imputation.
    XGBoost handles NaN at split points natively; imputing would discard
    the missingness signal that tells the model 'no lab drawn this hour'.
    """
    id_cols    = ["icustay_id", "window_start_time", "window_end_time"]
    label_cols = list(LABEL_HORIZONS.keys())
    feat_cols  = [f"{s}_{t}" for s in SIGNALS
                  for t in ("mean", "min", "max", "last")]
    return df_raw[id_cols + label_cols + feat_cols].copy()


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_fold_stats(df: pd.DataFrame, fold_df: pd.DataFrame,
                     cohort: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("FOLD STATISTICS")
    print("=" * 62)
    df_w = df.merge(fold_df, on="icustay_id", how="left")
    pts_fold = (
        cohort[["icustay_id", "sepsis_label"]]
        .merge(fold_df, on="icustay_id", how="left")
    )
    print(f"  {'Fold':>4}  {'Patients':>8}  {'Septic pts':>10}  "
          f"{'Total windows':>13}  {'Pos windows (6h)':>16}")
    for fold_id in range(N_FOLDS):
        pts_k = pts_fold[pts_fold["fold"] == fold_id]
        win_k = df_w[df_w["fold"] == fold_id]
        n_pts    = len(pts_k)
        n_septic = int(pts_k["sepsis_label"].sum())
        n_win    = len(win_k)
        n_pos    = int(win_k["label_6h"].sum())
        print(f"  {fold_id:>4}  {n_pts:>8}  {n_septic:>10}  "
              f"{n_win:>13,}  {n_pos:>16,}")


def print_null_rates(lstm_df: pd.DataFrame, xgb_df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("NULL RATES AFTER PREPROCESSING")
    print("=" * 62)
    n = len(lstm_df)

    print(f"\n  LSTM _last columns (post forward-fill + median imputation):")
    last_nulls = []
    for sig in SIGNALS:
        col = f"{sig}_last"
        pct = lstm_df[col].isna().sum() / n * 100
        last_nulls.append(pct)
        star = " **" if pct > 0.01 else ""
        print(f"    {col:<25}: {pct:6.2f}%{star}")
    print(f"    Mean null rate (_last): {np.mean(last_nulls):.4f}%")

    _fresh_expected = {
        "heart_rate": (85, 95), "dbp": (85, 95), "sbp": (85, 95),
        "resp_rate":  (85, 95), "spo2": (85, 95), "map": (85, 95),
        "temperature": (20, 40),
        "bilirubin":  (5, 12), "creatinine": (5, 12), "lactate": (5, 12),
        "platelets":  (5, 12), "wbc": (5, 12),
    }
    print(f"\n  LSTM _fresh columns (1 = observed in this window, pre-fill):")
    for sig in SIGNALS:
        col = f"{sig}_fresh"
        fresh_pct = (lstm_df[col] == 1).sum() / n * 100
        lo, hi = _fresh_expected.get(sig, (0, 100))
        flag = " **UNEXPECTED**" if not (lo <= fresh_pct <= hi) else ""
        print(f"    {col:<25}: {fresh_pct:6.2f}%{flag}")

    print(f"\n  XGBoost feature columns (no imputation applied):")
    xgb_feat = [c for c in xgb_df.columns
                if any(c.endswith(f"_{t}") for t in ("mean", "min", "max", "last"))]
    by_stat: dict[str, list[float]] = {t: [] for t in ("mean", "min", "max", "last")}
    for col in xgb_feat:
        stat = col.rsplit("_", 1)[1]
        pct  = xgb_df[col].isna().sum() / n * 100
        by_stat[stat].append(pct)
    for stat, rates in by_stat.items():
        print(f"    *_{stat:<6}: mean null {np.mean(rates):6.2f}%  "
              f"(min {np.min(rates):.1f}%  max {np.max(rates):.1f}%)")


def print_column_lists(lstm_df: pd.DataFrame, xgb_df: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("OUTPUT COLUMN LISTS")
    print("=" * 62)
    feat_lstm = [c for c in lstm_df.columns
                 if c.endswith("_last") or c.endswith("_fresh")]
    feat_xgb  = [c for c in xgb_df.columns
                 if any(c.endswith(f"_{t}") for t in ("mean", "min", "max", "last"))]
    print(f"  lstm_features.parquet: {len(lstm_df.columns)} columns "
          f"({len(feat_lstm)} features)")
    for c in lstm_df.columns:
        print(f"    {c}")
    print(f"\n  xgb_features.parquet: {len(xgb_df.columns)} columns "
          f"({len(feat_xgb)} features)")
    for c in xgb_df.columns:
        print(f"    {c}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────
    if not COHORT_PATH.exists():
        sys.exit(f"ERROR: {COHORT_PATH} not found. Run step 01 first.")

    cohort = pd.read_parquet(COHORT_PATH)
    df = load_stream_output()

    # ── Step 2: Drop *_stddev, *_slope ────────────────────────────────────
    df = drop_noisy_columns(df)

    # ── Step 3: Multi-horizon labels ──────────────────────────────────────
    print("\nComputing multi-horizon labels:")
    df = attach_multi_horizon_labels(df, cohort)

    # ── Step 4: Fold splits ────────────────────────────────────────────────
    print(f"\nBuilding {N_FOLDS}-fold stratified CV splits (random_state={RANDOM_STATE})...")
    fold_df = build_folds(cohort)
    fold_path = OUT_DIR / "folds.parquet"
    fold_df.to_parquet(fold_path, index=False)
    print(f"  Saved: {fold_path}")

    # ── Step 5: LSTM features ─────────────────────────────────────────────
    print("\nBuilding LSTM feature set...")
    lstm_df = build_lstm_features(df, fold_df)
    lstm_path = OUT_DIR / "lstm_features.parquet"
    lstm_df.to_parquet(lstm_path, index=False)
    print(f"  Saved: {lstm_path}  ({len(lstm_df):,} rows × {len(lstm_df.columns)} cols)")

    # ── Step 6: XGBoost features ──────────────────────────────────────────
    print("\nBuilding XGBoost feature set...")
    xgb_df = build_xgb_features(df)
    xgb_path = OUT_DIR / "xgb_features.parquet"
    xgb_df.to_parquet(xgb_path, index=False)
    print(f"  Saved: {xgb_path}  ({len(xgb_df):,} rows × {len(xgb_df.columns)} cols)")

    # ── Diagnostics ───────────────────────────────────────────────────────
    print_fold_stats(df, fold_df, cohort)
    print_null_rates(lstm_df, xgb_df)
    print_column_lists(lstm_df, xgb_df)

    print("\n" + "=" * 62)
    print("Step 03b complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()
