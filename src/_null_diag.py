"""Quick null-rate diagnostic on stream_output parquet."""
import glob
import pathlib

import pandas as pd

SIGNALS = [
    "bilirubin", "creatinine", "dbp", "heart_rate",
    "lactate", "map", "platelets", "resp_rate",
    "sbp", "spo2", "temperature", "wbc",
]
STATS = ["mean", "stddev", "min", "max", "last", "slope"]

parts = glob.glob("data/stream_output/**/*.parquet", recursive=True)
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
n = len(df)
print(f"Rows: {n:,}   Columns: {df.shape[1]}")

feat_cols = [f"{s}_{t}" for s in SIGNALS for t in STATS]

rows = []
for col in feat_cols:
    null_n = int(df[col].isna().sum())
    null_pct = null_n / n * 100
    sig = col.rsplit("_", 1)[0]
    stat = col.rsplit("_", 1)[1]
    rows.append({"feature": col, "signal": sig, "stat": stat,
                 "null_count": null_n, "null_pct": round(null_pct, 2)})

null_df = pd.DataFrame(rows).sort_values("null_pct", ascending=False)

out = pathlib.Path("data/processed/null_rates.csv")
null_df.to_csv(out, index=False)

print(null_df.to_string(index=False))
print()

# --- _last vs _mean comparison (the key forward-fill diagnostic) ---
print("=== _last vs _mean null rates by signal ===")
hdr = f"  {'signal':<15} {'_mean %':>8}  {'_last %':>8}  {'delta':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for sig in SIGNALS:
    m = null_df.loc[null_df["feature"] == f"{sig}_mean",  "null_pct"].iloc[0]
    lv = null_df.loc[null_df["feature"] == f"{sig}_last", "null_pct"].iloc[0]
    delta = lv - m
    marker = " <-- ffill would help" if lv > m + 0.1 else (" <-- ffill active" if lv < m - 0.1 else "")
    print(f"  {sig:<15} {m:>8.2f}  {lv:>8.2f}  {delta:>+8.2f}{marker}")

print()

# --- aggregate by stat ---
print("=== mean null % by stat type ===")
by_stat = null_df.groupby("stat")["null_pct"].mean().sort_values(ascending=False)
for stat, pct in by_stat.items():
    print(f"  {stat:<8}: {pct:.2f}%")

print()
print(f"Saved: {out}")
