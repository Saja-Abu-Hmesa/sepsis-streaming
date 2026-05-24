"""
04_train_xgboost.py  —  XGBoost training with 5-fold patient-level CV.

Trains one model per label horizon (6h / 12h / 24h).
Adds window_idx and window_idx_norm as temporal context features.
Uses early stopping + regularization to prevent overfitting on this small dataset.

Techniques for high AUPRC on 99% negative imbalanced data:
  - scale_pos_weight calibrated per fold
  - eval_metric='aucpr' drives early stopping toward the right objective
  - Temporal features (window_idx_norm) let trees distinguish early vs late ICU windows
  - Regularization: gamma, min_child_weight, alpha, lambda prevent tree depth bloat
  - Optuna search (--tune) explores the hyperparameter space efficiently

Usage:
  python src/04_train_xgboost.py           # default params
  python src/04_train_xgboost.py --tune    # Optuna search (50 trials, ~5 min)

Outputs:
  data/processed/oof_xgb.parquet           — out-of-fold predictions (all folds)
  data/processed/models/xgb_label_6h.json
  data/processed/models/xgb_label_12h.json
  data/processed/models/xgb_label_24h.json
"""

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import xgboost as xgb
except ImportError:
    sys.exit("ERROR: xgboost not installed. Run: pip install xgboost")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = DATA_DIR / "models"

LABELS = ["label_6h", "label_12h", "label_24h"]
SEED   = 42
N_FOLDS = 5

# ── Hyperparameters pre-tuned for this dataset size/imbalance ────────────────

DEFAULT_PARAMS = {
    "objective":             "binary:logistic",
    "eval_metric":           "aucpr",
    "tree_method":           "hist",
    "max_depth":             4,
    "learning_rate":         0.03,
    "n_estimators":          2000,
    "min_child_weight":      10,
    "subsample":             0.80,
    "colsample_bytree":      0.70,
    "reg_alpha":             0.10,
    "reg_lambda":            2.00,
    "gamma":                 1.00,
    "early_stopping_rounds": 50,
    "random_state":          SEED,
    "verbosity":             0,
}


# ── Feature engineering ───────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["icustay_id", "window_start_time"]).copy()
    df["window_idx"] = df.groupby("icustay_id").cumcount()
    stay_len = df.groupby("icustay_id")["window_idx"].transform("max").clip(lower=1)
    df["window_idx_norm"] = df["window_idx"] / stay_len
    # Hours into stay (from first window)
    first_ts = df.groupby("icustay_id")["window_start_time"].transform("min")
    df["hours_in_icu"] = (df["window_start_time"] - first_ts).dt.total_seconds() / 3600.0
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"icustay_id", "window_start_time", "window_end_time"} | set(LABELS)
    return [c for c in df.columns if c not in drop]


# ── Training ──────────────────────────────────────────────────────────────────

def sensitivity_at_specificity(y_true, y_score, target_spec=0.90):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    if not mask.any():
        return 0.0
    return float(tpr[mask][-1])


def train_fold(X_tr, y_tr, X_va, y_va, params: dict):
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    m = xgb.XGBClassifier(**{**params, "scale_pos_weight": spw})
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds = m.predict_proba(X_va)[:, 1]
    return m, preds


def run_cv(df: pd.DataFrame, fold_df: pd.DataFrame,
           label: str, params: dict) -> tuple[pd.Series, list, dict]:
    feat_cols = get_feature_cols(df)
    oof = pd.Series(np.nan, index=df.index, name=f"xgb_{label}")
    models = []
    fold_times: dict[int, float] = {}

    for k in range(N_FOLDS):
        tr_ids = set(fold_df.loc[fold_df["fold"] != k, "icustay_id"])
        va_ids = set(fold_df.loc[fold_df["fold"] == k,  "icustay_id"])

        tr = df["icustay_id"].isin(tr_ids)
        va = df["icustay_id"].isin(va_ids)

        X_tr = df.loc[tr, feat_cols].values
        y_tr = df.loc[tr, label].values.astype(np.float32)
        X_va = df.loc[va, feat_cols].values
        y_va = df.loc[va, label].values.astype(np.float32)

        fold_start = time.perf_counter()
        model, preds = train_fold(X_tr, y_tr, X_va, y_va, params)
        fold_times[k] = time.perf_counter() - fold_start

        oof.loc[va] = preds
        models.append(model)

        auroc = roc_auc_score(y_va, preds)
        auprc = average_precision_score(y_va, preds)
        s90   = sensitivity_at_specificity(y_va, preds, 0.90)
        pos   = int(y_va.sum())
        best  = getattr(model, "best_iteration", params["n_estimators"])
        print(f"  Fold {k}  pos={pos:>3}  AUROC={auroc:.4f}  "
              f"AUPRC={auprc:.4f}  Sens@90%Sp={s90:.4f}  trees={best}  "
              f"time={fold_times[k]:.1f}s")

    return oof, models, fold_times


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def optuna_tune(df: pd.DataFrame, fold_df: pd.DataFrame,
                label: str, n_trials: int = 50) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  optuna not found — pip install optuna. Using default params.")
        return DEFAULT_PARAMS.copy()

    feat_cols = get_feature_cols(df)
    tr_ids = set(fold_df.loc[fold_df["fold"] != 0, "icustay_id"])
    va_ids = set(fold_df.loc[fold_df["fold"] == 0, "icustay_id"])
    X_tr = df.loc[df["icustay_id"].isin(tr_ids), feat_cols].values
    y_tr = df.loc[df["icustay_id"].isin(tr_ids), label].values.astype(np.float32)
    X_va = df.loc[df["icustay_id"].isin(va_ids), feat_cols].values
    y_va = df.loc[df["icustay_id"].isin(va_ids), label].values.astype(np.float32)
    spw  = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    def objective(trial):
        p = {
            "objective":             "binary:logistic",
            "eval_metric":           "aucpr",
            "tree_method":           "hist",
            "max_depth":             trial.suggest_int("max_depth", 3, 7),
            "learning_rate":         trial.suggest_float("lr", 0.01, 0.1, log=True),
            "n_estimators":          2000,
            "min_child_weight":      trial.suggest_int("mcw", 3, 30),
            "subsample":             trial.suggest_float("sub", 0.6, 1.0),
            "colsample_bytree":      trial.suggest_float("col", 0.5, 1.0),
            "reg_alpha":             trial.suggest_float("alpha", 1e-3, 2.0, log=True),
            "reg_lambda":            trial.suggest_float("lam", 0.5, 5.0),
            "gamma":                 trial.suggest_float("gamma", 0.0, 3.0),
            "scale_pos_weight":      spw,
            "early_stopping_rounds": 50,
            "random_state":          SEED,
            "verbosity":             0,
        }
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return average_precision_score(y_va, m.predict_proba(X_va)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    bp = study.best_params
    print(f"  Optuna best AUPRC={study.best_value:.4f}  "
          f"max_depth={bp['max_depth']}  lr={bp['lr']:.4f}  mcw={bp['mcw']}")
    return {
        **DEFAULT_PARAMS,
        "max_depth":        bp["max_depth"],
        "learning_rate":    bp["lr"],
        "min_child_weight": bp["mcw"],
        "subsample":        bp["sub"],
        "colsample_bytree": bp["col"],
        "reg_alpha":        bp["alpha"],
        "reg_lambda":       bp["lam"],
        "gamma":            bp["gamma"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Optuna hyperparameter search (~5 min)")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df      = pd.read_parquet(DATA_DIR / "xgb_features.parquet")
    fold_df = pd.read_parquet(DATA_DIR / "folds.parquet")
    df      = add_temporal_features(df)

    n_feat = len(get_feature_cols(df))
    print(f"  {len(df):,} windows  {df['icustay_id'].nunique()} patients  {n_feat} features")

    oof_parts = [df[["icustay_id", "window_start_time", "window_end_time"] + LABELS].copy()]
    results   = {}
    all_fold_times: dict[str, dict[int, float]] = {}

    for label in LABELS:
        pos = int(df[label].sum())
        print(f"\n{'='*62}")
        print(f"XGBoost  label={label}  pos={pos:,} ({pos/len(df)*100:.2f}%)")
        print(f"{'='*62}")

        params = DEFAULT_PARAMS.copy()
        if args.tune:
            print(f"  Running Optuna search (n=50) for {label}...")
            params = optuna_tune(df, fold_df, label, n_trials=50)

        oof_preds, models, fold_times = run_cv(df, fold_df, label, params)
        all_fold_times[label] = fold_times
        oof_parts.append(oof_preds)

        y_true = df[label].values
        auroc  = roc_auc_score(y_true, oof_preds)
        auprc  = average_precision_score(y_true, oof_preds)
        s90    = sensitivity_at_specificity(y_true, oof_preds, 0.90)
        results[label] = {"AUROC": auroc, "AUPRC": auprc, "Sens@90Sp": s90}
        print(f"\n  OOF  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Sens@90%Sp={s90:.4f}")

        best_model_path = MODEL_DIR / f"xgb_{label}.json"
        models[0].save_model(str(best_model_path))
        print(f"  Saved: {best_model_path.name}")

    oof_df = pd.concat(oof_parts, axis=1)
    oof_df = oof_df.loc[:, ~oof_df.columns.duplicated()]
    oof_path = DATA_DIR / "oof_xgb.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"\nOOF predictions saved: {oof_path}")

    print(f"\n{'='*62}")
    print("XGBoost Results Summary")
    print(f"{'='*62}")
    print(f"  {'Label':<12}  {'AUROC':>8}  {'AUPRC':>8}  {'Sens@90%Sp':>11}")
    for lbl, m in results.items():
        print(f"  {lbl:<12}  {m['AUROC']:>8.4f}  {m['AUPRC']:>8.4f}  {m['Sens@90Sp']:>11.4f}")

    # ── Save training timing ──────────────────────────────────────────────────
    per_fold_flat = {
        f"{lbl}_fold{k}": round(t, 3)
        for lbl, times in all_fold_times.items()
        for k, t in times.items()
    }
    total_seconds = sum(per_fold_flat.values())
    timing = {
        "per_fold_seconds": per_fold_flat,
        "total_seconds":    round(total_seconds, 3),
        "labels_trained":   LABELS,
    }
    timing_path = DATA_DIR / "training_time_xgboost.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    mean_fold = total_seconds / (N_FOLDS * len(LABELS))
    print(f"\nTraining timing: total={total_seconds:.1f}s  mean_per_fold={mean_fold:.1f}s")
    print(f"Timing saved: {timing_path}")
    print("\nStep 04 complete.")


if __name__ == "__main__":
    main()
