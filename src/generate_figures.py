"""
generate_figures.py  —  Generates all paper figures from OOF predictions.

Produces the same outputs as the Jupyter notebook but runs as a plain script.
Figures saved to data/processed/paper_figures/ at 300 dpi.

Usage:
  python src/generate_figures.py
"""

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

ROOT    = pathlib.Path(__file__).resolve().parent.parent
DATA    = ROOT / "data" / "processed"
MODELS  = DATA / "models"
FIG_DIR = DATA / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELS   = ["label_6h", "label_12h", "label_24h"]
HORIZONS = {"label_6h": "6 h", "label_12h": "12 h", "label_24h": "24 h"}
SEED     = 42

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})
PALETTE = {
    "XGBoost":  "#1f77b4",
    "LSTM":     "#ff7f0e",
    "Ensemble": "#2ca02c",
    "Random":   "#aaaaaa",
}
MODELS_LIST = [("XGBoost", "xgb_"), ("LSTM", "lstm_"), ("Ensemble", "ens_")]


# ── Helpers ──────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=500):
    rng = np.random.default_rng(SEED)
    n   = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(metric_fn(yt, yp))
    return float(np.mean(scores)), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def sens_at_spec(y_true, y_score, target=0.90):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = (1 - fpr) >= target
    return float(tpr[mask][-1]) if mask.any() else 0.0


def compute_lead_times(df, score_col, threshold, cohort_df):
    septic = cohort_df[cohort_df["sepsis_label"] == 1][
        ["icustay_id", "sepsis_onset_time"]
    ].copy()
    if hasattr(septic["sepsis_onset_time"].dt, "tz") and septic["sepsis_onset_time"].dt.tz:
        septic["sepsis_onset_time"] = septic["sepsis_onset_time"].dt.tz_localize(None)
    mg = df.merge(septic, on="icustay_id", how="inner")
    mg = mg[mg["window_end_time"] <= mg["sepsis_onset_time"]]
    lead = []
    for _, grp in mg.groupby("icustay_id"):
        alarms = grp[grp[score_col] >= threshold]
        if alarms.empty:
            continue
        h = (grp["sepsis_onset_time"].iloc[0] - alarms["window_end_time"].min()
             ).total_seconds() / 3600
        if h >= 0:
            lead.append(h)
    return lead


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
cohort  = pd.read_parquet(DATA / "cohort.parquet")
folds   = pd.read_parquet(DATA / "folds.parquet")
ens_oof = pd.read_parquet(DATA / "oof_ensemble.parquet")
xgb_oof = pd.read_parquet(DATA / "oof_xgb.parquet")
xgb_raw = pd.read_parquet(DATA / "xgb_features.parquet")

for df in [ens_oof, xgb_oof, cohort]:
    for col in df.select_dtypes("datetimetz").columns:
        df[col] = df[col].dt.tz_localize(None)

n_pts    = len(cohort)
n_septic = int(cohort["sepsis_label"].sum())
n_win    = len(xgb_oof)
print(f"  {n_pts} patients  {n_septic} septic  {n_win:,} windows")


# ── Table 1 ───────────────────────────────────────────────────────────────────

pts = cohort[["icustay_id", "sepsis_label"]].merge(folds, on="icustay_id", how="left")
win = xgb_oof[["icustay_id", "label_6h"]].merge(folds, on="icustay_id", how="left")

rows = []
for k in range(5):
    pf = pts[pts["fold"] == k]
    wf = win[win["fold"] == k]
    rows.append({
        "Fold": k, "Patients": len(pf),
        "Septic": int(pf["sepsis_label"].sum()),
        "Non-septic": int((pf["sepsis_label"] == 0).sum()),
        "Windows": len(wf),
        "Pos(6h)": int(wf["label_6h"].sum()),
        "Prev%": f"{wf['label_6h'].mean()*100:.2f}",
    })
rows.append({
    "Fold": "Total", "Patients": len(pts),
    "Septic": n_septic, "Non-septic": n_pts - n_septic,
    "Windows": len(win), "Pos(6h)": int(win["label_6h"].sum()),
    "Prev%": f"{win['label_6h'].mean()*100:.2f}",
})
tbl1 = pd.DataFrame(rows).set_index("Fold")
print("\nTABLE 1 — Dataset Statistics")
print("=" * 65)
print(tbl1.to_string())


# ── Table 2 ───────────────────────────────────────────────────────────────────

print("\nTABLE 2 — Model Performance (bootstrap n=500)")
print("=" * 100)
tbl2_rows = []
for lbl in LABELS:
    y_true = ens_oof[lbl].values
    for name, prefix in MODELS_LIST:
        col = f"{prefix}{lbl}"
        if col not in ens_oof.columns:
            continue
        yp = ens_oof[col].values
        auroc, a_lo, a_hi = bootstrap_ci(y_true, yp, roc_auc_score)
        auprc, p_lo, p_hi = bootstrap_ci(y_true, yp, average_precision_score)
        s90 = sens_at_spec(y_true, yp, 0.90)
        s95 = sens_at_spec(y_true, yp, 0.95)
        tbl2_rows.append({
            "Horizon": HORIZONS[lbl], "Model": name,
            "AUROC": f"{auroc:.3f}", "AUROC 95%CI": f"[{a_lo:.3f},{a_hi:.3f}]",
            "AUPRC": f"{auprc:.3f}", "AUPRC 95%CI": f"[{p_lo:.3f},{p_hi:.3f}]",
            "Sens@90%Sp": f"{s90:.3f}", "Sens@95%Sp": f"{s95:.3f}",
        })
        print(f"  {HORIZONS[lbl]:>5} {name:<10}: "
              f"AUROC={auroc:.3f}[{a_lo:.3f},{a_hi:.3f}]  "
              f"AUPRC={auprc:.3f}[{p_lo:.3f},{p_hi:.3f}]  "
              f"Sens@90%Sp={s90:.3f}")

tbl2 = pd.DataFrame(tbl2_rows)


# ── Figure 1 — ROC curves ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, lbl in zip(axes, LABELS):
    y_true = ens_oof[lbl].values
    ax.plot([0, 1], [0, 1], color=PALETTE["Random"], ls="--", lw=1, label="Random  AUC=0.500")
    for name, prefix in MODELS_LIST:
        col = f"{prefix}{lbl}"
        if col not in ens_oof.columns:
            continue
        fpr, tpr, _ = roc_curve(y_true, ens_oof[col].values)
        auc = roc_auc_score(y_true, ens_oof[col].values)
        ax.plot(fpr, tpr, color=PALETTE[name],
                lw=2.5 if name == "Ensemble" else 1.8,
                label=f"{name}  AUC={auc:.3f}")
    ax.set_title(f"Horizon: {HORIZONS[lbl]}", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", framealpha=0.9)
axes[0].set_ylabel("True Positive Rate")
fig.suptitle("ROC Curves — 5-Fold OOF", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_roc_curves.png")
plt.close(fig)
print("\nSaved: fig1_roc_curves.png")


# ── Figure 2 — PR curves ──────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, lbl in zip(axes, LABELS):
    y_true   = ens_oof[lbl].values
    baseline = y_true.mean()
    ax.axhline(baseline, color=PALETTE["Random"], ls="--", lw=1,
               label=f"Random  AP={baseline:.3f}")
    for name, prefix in MODELS_LIST:
        col = f"{prefix}{lbl}"
        if col not in ens_oof.columns:
            continue
        prec, rec, _ = precision_recall_curve(y_true, ens_oof[col].values)
        ap = average_precision_score(y_true, ens_oof[col].values)
        ax.plot(rec, prec, color=PALETTE[name],
                lw=2.5 if name == "Ensemble" else 1.8,
                label=f"{name}  AP={ap:.3f}")
    ax.set_title(f"Horizon: {HORIZONS[lbl]}", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.9)
axes[0].set_ylabel("Precision")
fig.suptitle("Precision-Recall Curves — 5-Fold OOF", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_pr_curves.png")
plt.close(fig)
print("Saved: fig2_pr_curves.png")


# ── Figure 3 — Feature importance ─────────────────────────────────────────────

LABELS_SET = {"icustay_id", "window_start_time", "window_end_time",
              "label_6h", "label_12h", "label_24h"}
raw_feat_cols = [c for c in xgb_raw.columns if c not in LABELS_SET]
feat_cols     = raw_feat_cols + ["window_idx", "window_idx_norm", "hours_in_icu"]
feat_map      = {f"f{i}": name for i, name in enumerate(feat_cols)}

m6 = xgb.XGBClassifier()
m6.load_model(str(MODELS / "xgb_label_6h.json"))
raw_scores = m6.get_booster().get_score(importance_type="gain")
fi_named   = {feat_map.get(k, k): v for k, v in raw_scores.items()}

_SIG_LABELS = {
    "dbp":         "DBP",
    "sbp":         "SBP",
    "map":         "MAP",
    "spo2":        "SpO₂",
    "wbc":         "WBC",
    "resp_rate":   "Resp Rate",
    "heart_rate":  "Heart Rate",
    "temperature": "Temperature",
    "lactate":     "Lactate",
    "platelets":   "Platelets",
    "bilirubin":   "Bilirubin",
    "creatinine":  "Creatinine",
}


def clean_name(n):
    rn = {"window_idx_norm": "Window position (norm)",
          "window_idx":      "Window index",
          "hours_in_icu":    "Hours in ICU"}
    if n in rn:
        return rn[n]
    parts = n.rsplit("_", 1)
    if len(parts) == 2:
        sig, stat = parts
        sig  = _SIG_LABELS.get(sig, sig.replace("_", " ").title())
        stat = {"mean": "Mean", "min": "Min", "max": "Max", "last": "Last obs."}.get(stat, stat)
        return f"{sig} ({stat})"
    return n

fi_df = (pd.DataFrame.from_dict(fi_named, orient="index", columns=["Gain"])
           .sort_values("Gain", ascending=False)
           .head(20))
fi_df.index  = [clean_name(n) for n in fi_df.index]
fi_df["Gain%"] = fi_df["Gain"] / fi_df["Gain"].sum() * 100

fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#e74c3c" if any(x in n.lower() for x in ["position", "icu", "index"])
          else "#1f77b4" for n in fi_df.index]
bars = ax.barh(fi_df.index[::-1], fi_df["Gain%"][::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Importance (% total gain)")
ax.set_title("XGBoost Feature Importance — 6-hour Prediction Horizon\n(red = temporal features)",
             fontweight="bold")
ax.set_axisbelow(True)
for bar, val in zip(bars[::-1], fi_df["Gain%"]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_feature_importance.png")
plt.close(fig)
print("Saved: fig3_feature_importance.png")
print("  Top 5:", fi_df.head(5)["Gain%"].round(1).to_dict())


# ── Figure 4 — Per-fold performance ───────────────────────────────────────────

fold_rows = []
for k in range(5):
    va_ids  = set(folds.loc[folds["fold"] == k, "icustay_id"])
    fold_df = ens_oof[ens_oof["icustay_id"].isin(va_ids)]
    for lbl in LABELS:
        yt = fold_df[lbl].values
        if yt.sum() == 0:
            continue
        for name, prefix in MODELS_LIST:
            col = f"{prefix}{lbl}"
            if col not in fold_df.columns:
                continue
            fold_rows.append({
                "Fold": f"Fold {k}", "Horizon": HORIZONS[lbl], "Model": name,
                "AUROC": roc_auc_score(yt, fold_df[col].values),
                "AUPRC": average_precision_score(yt, fold_df[col].values),
            })
fold_perf = pd.DataFrame(fold_rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
    sub = fold_perf[fold_perf["Horizon"] == "6 h"]
    xf  = np.arange(5)
    for i, (name, _) in enumerate(MODELS_LIST):
        vals = sub[sub["Model"] == name][metric].values
        ax.bar(xf + (i - 1) * 0.25, vals, width=0.25,
               color=PALETTE[name], label=name, edgecolor="white")
    ax.set_xticks(xf)
    ax.set_xticklabels([f"Fold {k}" for k in range(5)])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per Fold — 6h Horizon", fontweight="bold")
    if metric == "AUROC":
        ax.set_ylim(0.5, 1.0)
    ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_fold_performance.png")
plt.close(fig)
print("Saved: fig4_fold_performance.png")


# ── Figure 5 — Time-to-detection ──────────────────────────────────────────────

fpr, tpr, thresh = roc_curve(ens_oof["label_6h"].values, ens_oof["ens_label_6h"].values)
mask  = (1 - fpr) >= 0.90
thr_90 = float(thresh[mask][-1])
lead_times = compute_lead_times(ens_oof, "ens_label_6h", thr_90, cohort)
sorted_lt  = np.sort(lead_times)
cdf        = np.arange(1, len(sorted_lt) + 1) / len(sorted_lt)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(lead_times, bins=20, color=PALETTE["Ensemble"], edgecolor="white")
axes[0].axvline(np.median(lead_times), color="red", ls="--", lw=2,
                label=f"Median={np.median(lead_times):.1f}h")
axes[0].axvline(np.mean(lead_times), color="orange", ls=":", lw=2,
                label=f"Mean={np.mean(lead_times):.1f}h")
axes[0].set_xlabel("Lead Time Before Sepsis Onset (hours)")
axes[0].set_ylabel("Number of Patients")
axes[0].set_title(f"Time-to-Detection (threshold={thr_90:.3f} @ 90% Sp)",
                  fontweight="bold")
axes[0].legend()

axes[1].plot(sorted_lt, cdf, color=PALETTE["Ensemble"], lw=2)
axes[1].axhline(0.5, color="red", ls="--", lw=1.5, alpha=0.7,
                label=f"Median={np.percentile(lead_times,50):.1f}h")
axes[1].axhline(0.8, color="orange", ls=":", lw=1.5, alpha=0.7,
                label=f"80th pct={np.percentile(lead_times,80):.1f}h")
axes[1].set_xlabel("Lead Time (hours)")
axes[1].set_ylabel("Cumulative Fraction")
axes[1].set_title("CDF of Lead Time", fontweight="bold")
axes[1].set_xlim(left=0)
axes[1].legend()

fig.suptitle("Ensemble — Early Detection Lead Time (6h horizon)", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_time_to_detection.png")
plt.close(fig)
print("Saved: fig5_time_to_detection.png")
print(f"  Detected: {len(lead_times)}/{n_septic}  "
      f"Median={np.median(lead_times):.1f}h  "
      f"{np.mean(np.array(lead_times)>4)*100:.0f}% detected >4h early")


# ── Figure 6 — Class distribution ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
pos_c = {HORIZONS[l]: int(xgb_oof[l].sum()) for l in LABELS}
neg_c = {HORIZONS[l]: len(xgb_oof) - pos_c[HORIZONS[l]] for l in LABELS}
x = np.arange(3)
ax.bar(x, [neg_c[h] for h in HORIZONS.values()], width=0.7, label="Negative",
       color="#c7dce8", edgecolor="#1f77b4", lw=0.8)
ax.bar(x, [pos_c[h] for h in HORIZONS.values()], width=0.7, label="Positive",
       color="#e74c3c", alpha=0.85)
for xi, lbl in zip(x, LABELS):
    n_pos = pos_c[HORIZONS[lbl]]
    prev  = xgb_oof[lbl].mean() * 100
    ax.text(xi, n_pos + 200, f"{n_pos:,}\n({prev:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="#c0392b", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{h} horizon" for h in HORIZONS.values()])
ax.set_ylabel("Windows")
ax.set_title("Class Distribution per Prediction Horizon\n(28,890 total windows)",
             fontweight="bold")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_label_distribution.png")
plt.close(fig)
print("Saved: fig6_label_distribution.png")


# ── Figure 7 — Summary panel ──────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax_roc  = fig.add_subplot(gs[0, 0])
ax_pr   = fig.add_subplot(gs[0, 1])
ax_fi   = fig.add_subplot(gs[0, 2])
ax_fold = fig.add_subplot(gs[1, 0])
ax_hist = fig.add_subplot(gs[1, 1])
ax_cdf  = fig.add_subplot(gs[1, 2])

y_true = ens_oof["label_6h"].values
ax_roc.plot([0, 1], [0, 1], color=PALETTE["Random"], ls="--", lw=1)
for name, prefix in MODELS_LIST:
    col = f"{prefix}label_6h"
    if col not in ens_oof.columns:
        continue
    fpr2, tpr2, _ = roc_curve(y_true, ens_oof[col].values)
    auc2 = roc_auc_score(y_true, ens_oof[col].values)
    ax_roc.plot(fpr2, tpr2, color=PALETTE[name],
                lw=2.5 if name == "Ensemble" else 1.5,
                label=f"{name} ({auc2:.3f})")
ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
ax_roc.set_title("(a) ROC — 6 h", fontweight="bold")
ax_roc.legend(fontsize=8, loc="lower right")
ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)

baseline = y_true.mean()
ax_pr.axhline(baseline, color=PALETTE["Random"], ls="--", lw=1,
              label=f"Random ({baseline:.3f})")
for name, prefix in MODELS_LIST:
    col = f"{prefix}label_6h"
    if col not in ens_oof.columns:
        continue
    prec2, rec2, _ = precision_recall_curve(y_true, ens_oof[col].values)
    ap2 = average_precision_score(y_true, ens_oof[col].values)
    ax_pr.plot(rec2, prec2, color=PALETTE[name],
               lw=2.5 if name == "Ensemble" else 1.5,
               label=f"{name} ({ap2:.3f})")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title("(b) PR Curve — 6 h", fontweight="bold")
ax_pr.legend(fontsize=8, loc="upper right")
ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1)

top10 = fi_df.head(10)
colors_fi = ["#e74c3c" if any(x in n.lower() for x in ["position", "icu", "index"])
             else "#1f77b4" for n in top10.index]
ax_fi.barh(top10.index[::-1], top10["Gain%"][::-1],
           color=colors_fi[::-1], edgecolor="white")
ax_fi.set_xlabel("Importance (% gain)")
ax_fi.set_title("(c) Feature Importance", fontweight="bold")

sub6 = fold_perf[fold_perf["Horizon"] == "6 h"]
xf   = np.arange(5)
for i, (name, _) in enumerate(MODELS_LIST):
    vals = sub6[sub6["Model"] == name]["AUROC"].values
    ax_fold.bar(xf + (i - 1) * 0.25, vals, width=0.25,
                color=PALETTE[name], label=name, edgecolor="white")
ax_fold.set_xticks(xf)
ax_fold.set_xticklabels([f"F{k}" for k in range(5)])
ax_fold.set_ylabel("AUROC"); ax_fold.set_ylim(0.5, 1.0)
ax_fold.set_title("(d) AUROC per Fold — 6 h", fontweight="bold")
ax_fold.legend(fontsize=8)

ax_hist.hist(lead_times, bins=15, color=PALETTE["Ensemble"], edgecolor="white")
ax_hist.axvline(np.median(lead_times), color="red", ls="--", lw=2,
                label=f"Median={np.median(lead_times):.1f}h")
ax_hist.set_xlabel("Lead Time (h)"); ax_hist.set_ylabel("Patients")
ax_hist.set_title("(e) Time-to-Detection", fontweight="bold")
ax_hist.legend(fontsize=8)

ax_cdf.plot(sorted_lt, cdf, color=PALETTE["Ensemble"], lw=2)
ax_cdf.axhline(0.5, color="red", ls="--", lw=1.2, alpha=0.8,
               label=f"Median={np.percentile(lead_times,50):.1f}h")
ax_cdf.set_xlabel("Lead Time (h)"); ax_cdf.set_ylabel("Cum. Fraction")
ax_cdf.set_title("(f) CDF of Lead Time", fontweight="bold")
ax_cdf.set_xlim(left=0); ax_cdf.legend(fontsize=8)

fig.suptitle("Sepsis Early Warning — Streaming Pipeline Results (MIMIC-III)",
             fontsize=13, fontweight="bold", y=1.01)
fig.savefig(FIG_DIR / "fig7_summary_panel.png")
plt.close(fig)
print("Saved: fig7_summary_panel.png  <-- main paper figure")


# ── Final stats ───────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("KEY NUMBERS FOR THE PAPER")
print("=" * 65)
print(f"\nDataset")
print(f"  Patients          : {n_pts}  ({n_septic} septic, {n_pts-n_septic} non-septic)")
print(f"  ICU windows       : {n_win:,}")
print(f"  Signals           : 12 vitals/labs × 30-min sliding window")
print(f"  Features (XGB)    : 51  (48 raw + 3 temporal)")
print(f"  Features (LSTM)   : 25  (12 ffilled + 12 binary freshness + position)")
print(f"  CV strategy       : 5-fold patient-level stratified")

print(f"\nModel Performance (ensemble, 5-fold OOF)")
for lbl in LABELS:
    yt   = ens_oof[lbl].values
    yp   = ens_oof[f"ens_{lbl}"].values
    auroc = roc_auc_score(yt, yp)
    auprc = average_precision_score(yt, yp)
    s90   = sens_at_spec(yt, yp, 0.90)
    lift  = auprc / yt.mean()
    print(f"  {HORIZONS[lbl]:>5}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
          f"Sens@90%Sp={s90:.3f}  AUPRC lift={lift:.1f}x")

print(f"\nTime-to-Detection (ensemble 6h, threshold@90%Sp={thr_90:.3f})")
print(f"  Detected         : {len(lead_times)} / {n_septic} septic patients")
print(f"  Median lead time : {np.median(lead_times):.1f} h before onset")
print(f"  Mean lead time   : {np.mean(lead_times):.1f} h")
lt_arr = np.array(lead_times)
print(f"  Detected >4h     : {np.mean(lt_arr>4)*100:.0f}% of detected patients")
print(f"  Detected >8h     : {np.mean(lt_arr>8)*100:.0f}% of detected patients")

print(f"\nFigures saved to: {FIG_DIR}")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")
