"""
06_evaluate.py  —  Final evaluation, ensemble, and plots.

Loads OOF predictions from XGBoost (04) and LSTM (05), computes:
  - Per-model AUROC / AUPRC with 95% bootstrap CIs
  - Soft ensemble (XGB + LSTM average)
  - Sensitivity @ 90% and 95% specificity
  - Time-to-detection: median lead time before sepsis onset
  - ROC + PR curve plots → data/processed/plots/

Usage:
  python src/06_evaluate.py [--no-plots]

Outputs:
  data/processed/plots/roc_curves.png
  data/processed/plots/pr_curves.png
  data/processed/oof_ensemble.parquet  — merged OOF with ensemble scores
"""

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
PLOT_DIR     = DATA_DIR / "plots"

LABELS = ["label_6h", "label_12h", "label_24h"]
SEED   = 42


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(SEED)
    n   = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        yp  = y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(metric_fn(yt, yp))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), lo, hi


def sensitivity_at_specificity(y_true, y_score, target_spec=0.90):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    return float(tpr[mask][-1]) if mask.any() else 0.0


# ── Time-to-detection ────────────────────────────────────────────────────────

def time_to_detection(df: pd.DataFrame, score_col: str, threshold: float,
                      cohort: pd.DataFrame) -> float:
    """
    For each septic patient, find the first window where score >= threshold
    BEFORE sepsis onset. Return median lead time in hours across patients.
    """
    septic = cohort[cohort["sepsis_label"] == 1][["icustay_id", "sepsis_onset_time"]]
    df = df.merge(septic, on="icustay_id", how="inner")
    df = df[df["window_end_time"] <= df["sepsis_onset_time"]]

    lead_times = []
    for stay_id, grp in df.groupby("icustay_id"):
        alarms = grp[grp[score_col] >= threshold]
        if alarms.empty:
            continue
        first_alarm = alarms["window_end_time"].min()
        onset       = grp["sepsis_onset_time"].iloc[0]
        lead_h      = (onset - first_alarm).total_seconds() / 3600.0
        if lead_h >= 0:
            lead_times.append(lead_h)

    return float(np.median(lead_times)) if lead_times else 0.0


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc(curves: list[dict], label: str, out_path: pathlib.Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    for c in curves:
        ax.plot(c["fpr"], c["tpr"],
                label=f"{c['name']}  AUC={c['auroc']:.3f}", lw=2)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC Curve — {label}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(curves: list[dict], label: str, baseline: float, out_path: pathlib.Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(baseline, color="k", ls="--", lw=0.8, alpha=0.5,
               label=f"Random  AP={baseline:.3f}")
    for c in curves:
        ax.plot(c["recall"], c["precision"],
                label=f"{c['name']}  AP={c['auprc']:.3f}", lw=2)
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"Precision–Recall Curve — {label}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    xgb_path  = DATA_DIR / "oof_xgb.parquet"
    lstm_path = DATA_DIR / "oof_lstm.parquet"

    have_xgb  = xgb_path.exists()
    have_lstm = lstm_path.exists()

    if not have_xgb and not have_lstm:
        sys.exit("ERROR: No OOF files found. Run steps 04 and/or 05 first.")

    # Load and merge OOF files
    if have_xgb:
        xgb_df = pd.read_parquet(xgb_path)
        print(f"XGBoost OOF loaded:  {len(xgb_df):,} rows")
    if have_lstm:
        lstm_df = pd.read_parquet(lstm_path)
        print(f"LSTM OOF loaded:     {len(lstm_df):,} rows")

    if have_xgb and have_lstm:
        merge_on = ["icustay_id", "window_start_time", "window_end_time"] + LABELS
        df = xgb_df.merge(
            lstm_df[["icustay_id", "window_start_time", "window_end_time"]
                    + [f"lstm_{l}" for l in LABELS]],
            on=["icustay_id", "window_start_time", "window_end_time"],
            how="inner",
        )
        print(f"Merged OOF:          {len(df):,} rows")
    elif have_xgb:
        df = xgb_df
    else:
        df = lstm_df

    # Add ensemble scores
    for lbl in LABELS:
        cols = []
        if have_xgb  and f"xgb_{lbl}"  in df.columns: cols.append(f"xgb_{lbl}")
        if have_lstm and f"lstm_{lbl}" in df.columns: cols.append(f"lstm_{lbl}")
        if len(cols) > 1:
            df[f"ens_{lbl}"] = df[cols].mean(axis=1)

    # Save merged OOF
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ens_path = DATA_DIR / "oof_ensemble.parquet"
    df.to_parquet(ens_path, index=False)
    print(f"Ensemble OOF saved:  {ens_path}")

    # Load cohort for time-to-detection
    cohort_path = DATA_DIR / "cohort.parquet"
    cohort = pd.read_parquet(cohort_path) if cohort_path.exists() else None
    # Ensure timezone-naive timestamps for merge
    for col in ["window_start_time", "window_end_time"]:
        if col in df.columns and hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)

    print(f"\n{'='*72}")
    print("Evaluation Results")
    print(f"{'='*72}")

    for lbl in LABELS:
        y_true    = df[lbl].values
        pos_rate  = y_true.mean()
        if y_true.sum() == 0:
            print(f"\n  {lbl}: no positives — skipping")
            continue

        print(f"\n  Label: {lbl}  (positives={int(y_true.sum()):,}  {pos_rate*100:.2f}%)")
        print(f"  {'Model':<12}  {'AUROC':>8}  {'95% CI':>16}  {'AUPRC':>8}  {'95% CI':>16}  "
              f"{'Sens@90':>8}  {'Sens@95':>8}")

        curves_roc = []
        curves_pr  = []

        models_to_eval = []
        if have_xgb  and f"xgb_{lbl}"  in df.columns: models_to_eval.append(("XGBoost", f"xgb_{lbl}"))
        if have_lstm and f"lstm_{lbl}" in df.columns: models_to_eval.append(("LSTM",    f"lstm_{lbl}"))
        if f"ens_{lbl}" in df.columns:                models_to_eval.append(("Ensemble", f"ens_{lbl}"))

        for name, col in models_to_eval:
            y_pred = df[col].values

            auroc,  a_lo, a_hi = bootstrap_ci(y_true, y_pred, roc_auc_score)
            auprc,  p_lo, p_hi = bootstrap_ci(y_true, y_pred, average_precision_score)
            s90 = sensitivity_at_specificity(y_true, y_pred, 0.90)
            s95 = sensitivity_at_specificity(y_true, y_pred, 0.95)

            print(f"  {name:<12}  {auroc:>8.4f}  [{a_lo:.4f}, {a_hi:.4f}]  "
                  f"{auprc:>8.4f}  [{p_lo:.4f}, {p_hi:.4f}]  "
                  f"{s90:>8.4f}  {s95:>8.4f}")

            fpr, tpr, _         = roc_curve(y_true, y_pred)
            prec, rec, _        = precision_recall_curve(y_true, y_pred)
            curves_roc.append({"name": name, "fpr": fpr, "tpr": tpr, "auroc": auroc})
            curves_pr.append({"name": name, "recall": rec, "precision": prec, "auprc": auprc})

        # Time-to-detection for best model (ensemble if available, else XGB)
        if cohort is not None:
            best_col = f"ens_{lbl}" if f"ens_{lbl}" in df.columns else \
                       (f"xgb_{lbl}" if have_xgb else f"lstm_{lbl}")
            # Find threshold at 90% spec
            fpr, tpr, thresh = roc_curve(y_true, df[best_col].values)
            spec = 1 - fpr
            mask = spec >= 0.90
            thr_90 = float(thresh[mask][-1]) if mask.any() else 0.5
            # Ensure tz-naive timestamps in cohort
            if "sepsis_onset_time" in cohort.columns:
                if hasattr(cohort["sepsis_onset_time"].dt, "tz") and cohort["sepsis_onset_time"].dt.tz is not None:
                    cohort = cohort.copy()
                    cohort["sepsis_onset_time"] = cohort["sepsis_onset_time"].dt.tz_localize(None)
            lead_h = time_to_detection(df, best_col, thr_90, cohort)
            print(f"  Time-to-detection (threshold@90%Spec={thr_90:.3f}): "
                  f"median {lead_h:.1f} h before onset")

        if not args.no_plots:
            try:
                roc_out = PLOT_DIR / f"roc_{lbl}.png"
                pr_out  = PLOT_DIR / f"pr_{lbl}.png"
                plot_roc(curves_roc, lbl, roc_out)
                plot_pr(curves_pr, lbl, pos_rate, pr_out)
                print(f"  Plots saved: {roc_out.name}  {pr_out.name}")
            except Exception as e:
                print(f"  Plot error: {e}")

    print(f"\n{'='*72}")
    print("Step 06 complete.")


if __name__ == "__main__":
    main()
