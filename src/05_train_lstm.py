"""
05_train_lstm.py  —  Causal LSTM for multi-task sepsis early warning.

Architecture:
  - Input: 25 features/window (12 _last + 12 _fresh + window_idx_norm)
  - 2-layer unidirectional LSTM, hidden=64, dropout=0.4
  - 3 linear output heads: label_6h / label_12h / label_24h
  - Loss: focal loss (gamma=2) per head, summed

Why focal loss: 99% negative class — focal loss down-weights easy negatives so
the model focuses on the ambiguous windows near sepsis onset.

Why unidirectional: causal (no future leakage); consistent with real-time streaming.

Why per-fold z-score normalization: _last features have wildly different scales
(platelets ~50-400 vs spo2 ~80-100 vs lactate ~0-15). Normalizing with train-fold
statistics avoids leakage and stabilizes LSTM training.

Sequence handling:
  - Per patient: sort by window_start_time, take LAST max_seq_len windows
  - Pad shorter sequences at the START (pre-padding) so last window = last timestep
  - Loss and metrics only computed on non-padded positions

Usage:
  python src/05_train_lstm.py [--epochs 80] [--hidden 64] [--layers 2]

Outputs:
  data/processed/oof_lstm.parquet
  data/processed/models/lstm_fold{k}.pt   (k=0..4)
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

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    sys.exit("ERROR: torch not installed. Run: pip install torch")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = DATA_DIR / "models"

LABELS    = ["label_6h", "label_12h", "label_24h"]
SIGNALS   = ["bilirubin", "creatinine", "dbp", "heart_rate",
             "lactate", "map", "platelets", "resp_rate",
             "sbp", "spo2", "temperature", "wbc"]
LAST_COLS  = [f"{s}_last"  for s in SIGNALS]
FRESH_COLS = [f"{s}_fresh" for s in SIGNALS]
SEED       = 42
N_FOLDS    = 5
MAX_SEQ    = 300   # clip long stays at last 300 windows (~150 h at 30-min slide)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")   # CPU is fine: 136 patients, hidden=64


# ── Dataset ───────────────────────────────────────────────────────────────────

class PatientDataset(Dataset):
    """Each item is one patient's complete sequence (padded to max_seq in batch)."""

    def __init__(self, sequences: list[dict]):
        self.seqs = sequences

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def collate_fn(batch):
    """Pad batch to max length in batch; pre-pad (so last real step = last position)."""
    max_len = max(s["seq_len"] for s in batch)
    n_feat  = batch[0]["x"].shape[1]
    n_lbl   = batch[0]["y"].shape[1]

    X   = torch.zeros(len(batch), max_len, n_feat)
    Y   = torch.zeros(len(batch), max_len, n_lbl)
    M   = torch.zeros(len(batch), max_len, dtype=torch.bool)  # True = real
    L   = torch.zeros(len(batch), dtype=torch.long)

    for i, s in enumerate(batch):
        t = s["seq_len"]
        X[i, max_len - t:] = s["x"]
        Y[i, max_len - t:] = s["y"]
        M[i, max_len - t:] = True
        L[i]               = t

    return X.to(DEVICE), Y.to(DEVICE), M.to(DEVICE), L.to(DEVICE)


# ── Model ─────────────────────────────────────────────────────────────────────

class SepsisLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float, n_tasks: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop  = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_tasks)])

    def forward(self, x):
        out, _ = self.lstm(x)               # (B, T, H)
        out    = self.drop(out)
        logits = torch.cat([h(out) for h in self.heads], dim=-1)  # (B, T, n_tasks)
        return logits


# ── Loss ──────────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, mask, alpha=0.75, gamma=2.0):
    """
    Focal loss computed only on non-padded positions.
    alpha: weight for positive class (higher → more recall).
    gamma: focusing parameter (2 = standard, down-weights easy negatives).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt  = torch.exp(-bce)
    at  = targets * alpha + (1 - targets) * (1 - alpha)
    fl  = at * (1 - pt) ** gamma * bce          # (B, T, n_tasks)
    fl  = fl * mask.unsqueeze(-1).float()        # zero out padding
    return fl.sum() / mask.sum().clamp(min=1)


# ── Per-fold normalization ────────────────────────────────────────────────────

def compute_fold_stats(df: pd.DataFrame, fold_df: pd.DataFrame) -> dict:
    """Compute per-fold (mean, std) for _last features using TRAIN windows only."""
    stats = {}
    for k in range(N_FOLDS):
        tr_ids = set(fold_df.loc[fold_df["fold"] != k, "icustay_id"])
        tr     = df[df["icustay_id"].isin(tr_ids)]
        fold_stats = {}
        for col in LAST_COLS:
            mu  = float(tr[col].mean())
            std = float(tr[col].std())
            fold_stats[col] = (mu, max(std, 1e-6))
        stats[k] = fold_stats
    return stats


def normalize(arr: np.ndarray, col_names: list[str],
              fold_stats: dict[str, tuple]) -> np.ndarray:
    out = arr.copy().astype(np.float32)
    for i, col in enumerate(col_names):
        if col in fold_stats:
            mu, std = fold_stats[col]
            out[:, i] = (out[:, i] - mu) / std
    return out


# ── Build patient sequences ───────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, fold_df: pd.DataFrame,
                    fold_stats_all: dict) -> dict[int, list[dict]]:
    """
    Returns {fold_id: [patient_dict, ...]} for each fold.
    Each patient_dict: {'x': Tensor(T,25), 'y': Tensor(T,3),
                        'seq_len': T, 'icustay_id': int,
                        'window_start_time': array, 'window_end_time': array}
    """
    id_to_fold = fold_df.set_index("icustay_id")["fold"].to_dict()
    feat_cols  = LAST_COLS + FRESH_COLS + ["window_idx_norm"]
    n_feat     = len(feat_cols)

    fold_seqs: dict[int, list] = {k: [] for k in range(N_FOLDS)}

    for stay_id, grp in df.sort_values("window_start_time").groupby("icustay_id"):
        fold_id = id_to_fold.get(stay_id)
        if fold_id is None:
            continue

        grp = grp.sort_values("window_start_time")

        # Take last MAX_SEQ windows
        if len(grp) > MAX_SEQ:
            grp = grp.iloc[-MAX_SEQ:]

        x_raw = grp[feat_cols].values.astype(np.float32)
        y_raw = grp[LABELS].values.astype(np.float32)

        # Normalize _last columns using this fold's train stats
        x_norm = normalize(x_raw, feat_cols, fold_stats_all[fold_id])

        seq = {
            "x":                torch.tensor(x_norm),
            "y":                torch.tensor(y_raw),
            "seq_len":          len(grp),
            "icustay_id":       stay_id,
            "window_start_time": grp["window_start_time"].values,
            "window_end_time":   grp["window_end_time"].values,
        }
        fold_seqs[fold_id].append(seq)

    return fold_seqs


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for X, Y, M, _ in loader:
        optimizer.zero_grad()
        logits = model(X)
        loss   = focal_loss(logits, Y, M)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, seqs: list[dict]) -> tuple[float, float, np.ndarray]:
    """Evaluate on a list of patient sequences. Returns (AUROC, AUPRC, preds_6h)."""
    model.eval()
    all_true  = {lbl: [] for lbl in LABELS}
    all_pred  = {lbl: [] for lbl in LABELS}

    for s in seqs:
        x   = s["x"].unsqueeze(0).to(DEVICE)          # (1, T, F)
        logits = model(x).squeeze(0)                   # (T, n_tasks)
        probs  = torch.sigmoid(logits).cpu().numpy()   # (T, 3)
        y_true = s["y"].numpy()                        # (T, 3)

        for j, lbl in enumerate(LABELS):
            all_true[lbl].append(y_true[:, j])
            all_pred[lbl].append(probs[:, j])

    results = {}
    for j, lbl in enumerate(LABELS):
        yt = np.concatenate(all_true[lbl])
        yp = np.concatenate(all_pred[lbl])
        if yt.sum() == 0:
            results[lbl] = (0.5, 0.0)
        else:
            results[lbl] = (roc_auc_score(yt, yp), average_precision_score(yt, yp))

    return results


# ── OOF collection ────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_oof_predictions(model, seqs: list[dict]) -> list[dict]:
    model.eval()
    rows = []
    for s in seqs:
        x      = s["x"].unsqueeze(0).to(DEVICE)
        logits = model(x).squeeze(0)
        probs  = torch.sigmoid(logits).cpu().numpy()   # (T, 3)
        y_true = s["y"].numpy()                        # (T, 3)
        T      = s["seq_len"]
        for t in range(T):
            row = {
                "icustay_id":       s["icustay_id"],
                "window_start_time": s["window_start_time"][t],
                "window_end_time":   s["window_end_time"][t],
            }
            for j, lbl in enumerate(LABELS):
                row[lbl]             = int(y_true[t, j])
                row[f"lstm_{lbl}"]   = float(probs[t, j])
            rows.append(row)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int,   default=16)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df      = pd.read_parquet(DATA_DIR / "lstm_features.parquet")
    fold_df = pd.read_parquet(DATA_DIR / "folds.parquet")

    # Add temporal position feature
    df = df.sort_values(["icustay_id", "window_start_time"])
    df["window_idx"] = df.groupby("icustay_id").cumcount()
    stay_len = df.groupby("icustay_id")["window_idx"].transform("max").clip(lower=1)
    df["window_idx_norm"] = df["window_idx"] / stay_len

    print(f"  {len(df):,} windows  {df['icustay_id'].nunique()} patients")

    fold_stats = compute_fold_stats(df, fold_df)
    fold_seqs  = build_sequences(df, fold_df, fold_stats)

    total_seqs = sum(len(v) for v in fold_seqs.values())
    print(f"  Patient sequences built: {total_seqs} total  "
          f"(MAX_SEQ={MAX_SEQ}  n_feat=25)")

    n_feat  = 25   # 12 _last + 12 _fresh + window_idx_norm
    n_tasks = len(LABELS)

    all_oof_rows = []
    results = {}
    fold_times: dict[int, float] = {}

    for k in range(N_FOLDS):
        print(f"\n{'='*62}")
        print(f"LSTM Fold {k}")
        print(f"{'='*62}")
        fold_start = time.perf_counter()

        val_seqs   = fold_seqs[k]
        train_seqs = [s for j in range(N_FOLDS) if j != k for s in fold_seqs[j]]

        print(f"  train={len(train_seqs)} patients  val={len(val_seqs)} patients")

        train_loader = DataLoader(
            PatientDataset(train_seqs),
            batch_size=args.batch,
            shuffle=True,
            collate_fn=collate_fn,
        )

        model = SepsisLSTM(n_feat, args.hidden, args.layers,
                           args.dropout, n_tasks).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        best_auprc  = -1.0
        best_state  = None
        patience    = 20
        no_improve  = 0

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer)
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                val_metrics = eval_epoch(model, val_seqs)
                auprc_6h    = val_metrics["label_6h"][1]
                auroc_6h    = val_metrics["label_6h"][0]
                print(f"  ep{epoch:>3}  loss={loss:.4f}  "
                      f"AUROC_6h={auroc_6h:.4f}  AUPRC_6h={auprc_6h:.4f}")

                if auprc_6h > best_auprc:
                    best_auprc = auprc_6h
                    best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 5
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        # Restore best weights and save
        if best_state is not None:
            model.load_state_dict(best_state)
        model_path = MODEL_DIR / f"lstm_fold{k}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "args": vars(args),
                    "n_feat": n_feat,
                    "n_tasks": n_tasks}, str(model_path))
        print(f"  Saved: {model_path.name}  (best AUPRC_6h={best_auprc:.4f})")

        # Collect OOF
        oof_rows = collect_oof_predictions(model, val_seqs)
        all_oof_rows.extend(oof_rows)

        val_m = eval_epoch(model, val_seqs)
        fold_times[k] = time.perf_counter() - fold_start
        for lbl in LABELS:
            auroc, auprc = val_m[lbl]
            print(f"  {lbl:>10}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        print(f"  Fold time: {fold_times[k]:.1f}s")
        results[k] = val_m

    # OOF summary across all folds
    oof_df = pd.DataFrame(all_oof_rows)
    oof_path = DATA_DIR / "oof_lstm.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"\nOOF predictions saved: {oof_path}  ({len(oof_df):,} rows)")

    print(f"\n{'='*62}")
    print("LSTM Results Summary (OOF across all folds)")
    print(f"{'='*62}")
    print(f"  {'Label':<12}  {'AUROC':>8}  {'AUPRC':>8}")
    for lbl in LABELS:
        yt = oof_df[lbl].values
        yp = oof_df[f"lstm_{lbl}"].values
        if yt.sum() == 0:
            continue
        print(f"  {lbl:<12}  {roc_auc_score(yt, yp):>8.4f}  "
              f"{average_precision_score(yt, yp):>8.4f}")

    # ── Save training timing ──────────────────────────────────────────────────
    per_fold_flat = {f"fold{k}": round(t, 3) for k, t in fold_times.items()}
    total_seconds = sum(per_fold_flat.values())
    timing = {
        "per_fold_seconds": per_fold_flat,
        "total_seconds":    round(total_seconds, 3),
        "labels_trained":   LABELS,
    }
    timing_path = DATA_DIR / "training_time_lstm.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    mean_fold = total_seconds / N_FOLDS if N_FOLDS > 0 else 0
    print(f"\nTraining timing: total={total_seconds:.1f}s  mean_per_fold={mean_fold:.1f}s")
    print(f"Timing saved: {timing_path}")
    print("\nStep 05 complete.")


if __name__ == "__main__":
    main()
