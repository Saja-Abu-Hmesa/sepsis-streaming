"""
02_build_stream_input.py  —  Build chunked CSV stream-input for Spark Structured Streaming.

Reads CHARTEVENTS (vitals) and LABEVENTS (key labs), unifies them into a
single long table, sorts globally by charttime, then splits into ~50 CSV
files so Spark file-source picks them up incrementally.

Output schema (each CSV):
  icustay_id   int
  charttime    string  (ISO-8601, "YYYY-MM-DD HH:MM:SS")
  signal_name  string  (canonical, e.g. "heart_rate")
  valuenum     float
  source_table string  ("chartevents" | "labevents")

Usage:
    python src/02_build_stream_input.py
    python src/02_build_stream_input.py --data-dir data/raw \\
        --cohort data/processed/cohort.parquet \\
        --out-dir data/stream_input --n-files 50
"""

import argparse
import json
import pathlib
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Signal-to-itemid maps ─────────────────────────────────────────────────────
# Each entry: signal_name -> {carevue: [...], metavision: [...], labevents: [...]}
# Temperature items tagged with 'F' need Fahrenheit->Celsius conversion.

SIGNAL_ITEMIDS = {
    "heart_rate": {
        "carevue":    [211],
        "metavision": [220045],
    },
    "sbp": {
        "carevue":    [51, 442, 455, 6701],
        "metavision": [220050, 220179, 225309, 227243],
    },
    "dbp": {
        "carevue":    [8368, 8440, 8441, 8503, 8504, 8506],
        "metavision": [220051, 220180, 225310, 227242],
    },
    "map": {
        "carevue":    [52, 456],
        "metavision": [220052, 220181],
    },
    "resp_rate": {
        "carevue":    [615, 618],
        "metavision": [220210, 224690],
    },
    "temperature": {
        # mixed C and F — see TEMP_F_ITEMIDS for conversion
        "carevue":    [676, 677, 678, 679],
        "metavision": [223761, 223762],
    },
    "spo2": {
        "carevue":    [646],
        "metavision": [220277],
    },
    "lactate": {
        "labevents": [50813],
    },
    "wbc": {
        "labevents": [51300, 51301],
    },
    "creatinine": {
        "labevents": [50912],
    },
    "bilirubin": {
        "labevents": [50885],
    },
    "platelets": {
        "labevents": [51265],
    },
}

# Fahrenheit itemids — must be converted to Celsius
TEMP_F_ITEMIDS = {678, 679, 223761}

# Physiological plausibility bounds (inclusive).  Values outside are dropped.
PHYS_BOUNDS = {
    "heart_rate":  (20,   300),
    "sbp":         (40,   300),
    "dbp":         (10,   200),
    "map":         (20,   200),
    "resp_rate":   (4,    70),
    "temperature": (25.0, 45.0),   # Celsius
    "spo2":        (50,   100),
    "lactate":     (0.1,  30.0),
    "wbc":         (0.1,  500.0),
    "creatinine":  (0.1,  30.0),
    "bilirubin":   (0.1,  80.0),
    "platelets":   (1,    2000),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: pathlib.Path, **kw) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: required file not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False, **kw)
    if df.empty:
        print(f"WARNING: {path.name} loaded empty")
    return df


def _itemid_to_signal() -> dict[int, tuple[str, str]]:
    """Returns {itemid: (signal_name, source_tag)} for all chart itemids."""
    m: dict[int, tuple[str, str]] = {}
    for sig, sources in SIGNAL_ITEMIDS.items():
        for src, ids in sources.items():
            if src == "labevents":
                continue
            for iid in ids:
                m[iid] = (sig, src)
    return m


def _lab_itemid_to_signal() -> dict[int, str]:
    """Returns {itemid: signal_name} for all lab itemids."""
    m: dict[int, str] = {}
    for sig, sources in SIGNAL_ITEMIDS.items():
        for iid in sources.get("labevents", []):
            m[iid] = sig
    return m


# ── Main processing ───────────────────────────────────────────────────────────

def build_vitals(ce_path: pathlib.Path,
                 cohort_ids: set,
                 iid_map: dict) -> pd.DataFrame:
    """
    Load and filter CHARTEVENTS for vitals.
    Returns long DataFrame with columns:
      icustay_id, charttime, signal_name, valuenum, source_table, _src_tag
    """
    all_ce_ids = set(iid_map.keys())

    print("  Reading CHARTEVENTS in chunks…")
    chunks = []
    total_read = 0
    for chunk in pd.read_csv(ce_path, chunksize=100_000, low_memory=False):
        total_read += len(chunk)
        sub = chunk[
            chunk.itemid.isin(all_ce_ids) &
            chunk.icustay_id.isin(cohort_ids)
        ][["icustay_id", "itemid", "charttime", "valuenum"]].copy()
        if len(sub):
            chunks.append(sub)

    if not chunks:
        print("  ERROR: no CHARTEVENTS rows matched", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  CHARTEVENTS read: {total_read:,} total rows -> "
          f"{len(df):,} matching cohort+itemid filter")

    # null valuenum
    n_null = df.valuenum.isna().sum()
    df = df.dropna(subset=["valuenum", "icustay_id"])
    print(f"  Dropped {n_null:,} rows with null valuenum")

    # Temperature F -> C
    f_mask = df.itemid.isin(TEMP_F_ITEMIDS)
    df.loc[f_mask, "valuenum"] = (df.loc[f_mask, "valuenum"] - 32) * 5 / 9
    if f_mask.sum():
        print(f"  Converted {f_mask.sum():,} temperature values F->C")

    # Map itemid -> signal_name
    df["signal_name"] = df.itemid.map(lambda x: iid_map[x][0])
    df["_src_tag"]    = df.itemid.map(lambda x: iid_map[x][1])
    df["source_table"] = "chartevents"

    # Coverage report: rows per signal × source
    print("  Coverage per signal (chartevents):")
    cov = df.groupby(["signal_name", "_src_tag"]).size().reset_index(name="rows")
    for sig in sorted(df.signal_name.unique()):
        rows = cov[cov.signal_name == sig]
        detail = ", ".join(
            f"{r['_src_tag']}={r['rows']:,}" for _, r in rows.iterrows()
        )
        print(f"    {sig:15s}: {df[df.signal_name==sig].valuenum.count():>7,} rows  [{detail}]")

    df = df.drop(columns=["itemid", "_src_tag"])
    df["icustay_id"] = df["icustay_id"].astype(int)
    return df[["icustay_id", "charttime", "signal_name", "valuenum", "source_table"]]


def build_labs(lab_path: pathlib.Path,
               icu: pd.DataFrame,
               cohort_ids: set,
               lab_iid_map: dict) -> pd.DataFrame:
    """
    Load and filter LABEVENTS for key labs.
    Assigns icustay_id by matching lab charttime to ICU stay [intime, outtime].
    Returns long DataFrame with same schema as vitals.
    """
    all_lab_ids = set(lab_iid_map.keys())

    lab = _load(lab_path)
    # filter to our itemids and to cohort hadm_ids
    cohort_hadm = set(icu[icu.icustay_id.isin(cohort_ids)].hadm_id.dropna())
    lab = lab[lab.itemid.isin(all_lab_ids) & lab.hadm_id.isin(cohort_hadm)].copy()
    print(f"  LABEVENTS matched itemid+hadm_id: {len(lab):,} rows")

    n_null = lab.valuenum.isna().sum()
    lab = lab.dropna(subset=["valuenum", "hadm_id", "charttime"])
    print(f"  Dropped {n_null:,} rows with null valuenum")

    lab["charttime"] = pd.to_datetime(lab["charttime"])
    icu_work = icu[icu.icustay_id.isin(cohort_ids)].copy()
    icu_work["intime"]  = pd.to_datetime(icu_work["intime"])
    icu_work["outtime"] = pd.to_datetime(icu_work["outtime"])

    # Assign icustay_id: for each lab row find the stay where the lab time
    # falls within [intime − 6 h, outtime + 6 h].  Grace period handles
    # labs drawn just before/after ICU stay boundaries.
    # If multiple stays match, keep the one with smallest |charttime − intime|.
    hadm_groups = icu_work.groupby("hadm_id")
    assigned_rows = []
    n_unmatched = 0

    for hadm_id, lab_grp in lab.groupby("hadm_id"):
        if hadm_id not in hadm_groups.groups:
            n_unmatched += len(lab_grp)
            continue
        stays = hadm_groups.get_group(hadm_id)
        for _, row in lab_grp.iterrows():
            t = row["charttime"]
            # stays covering this time (with 6h grace)
            cover = stays[
                (stays.intime  - pd.Timedelta("6h") <= t) &
                (stays.outtime + pd.Timedelta("6h") >= t)
            ]
            if cover.empty:
                # fall back: nearest stay in same admission
                cover = stays.copy()
                cover = cover.iloc[
                    [(cover.intime - t).abs().values.argmin()]
                ]
            # pick stay closest to lab time
            best = cover.iloc[
                [(cover.intime - t).abs().values.argmin()]
            ].iloc[0]
            new_row = row.to_dict()
            new_row["icustay_id"] = int(best["icustay_id"])
            assigned_rows.append(new_row)

    if not assigned_rows:
        print("  WARNING: no lab rows could be assigned to an ICU stay")
        return pd.DataFrame(
            columns=["icustay_id", "charttime", "signal_name",
                     "valuenum", "source_table"])

    if n_unmatched:
        print(f"  {n_unmatched:,} lab rows dropped — hadm_id not in ICU stays")

    lab_assigned = pd.DataFrame(assigned_rows)
    print(f"  Lab rows assigned to ICU stays: {len(lab_assigned):,}")

    lab_assigned["signal_name"]  = lab_assigned["itemid"].map(lab_iid_map)
    lab_assigned["source_table"] = "labevents"

    print("  Coverage per signal (labevents):")
    for sig in sorted(lab_assigned.signal_name.unique()):
        n = (lab_assigned.signal_name == sig).sum()
        print(f"    {sig:15s}: {n:>7,} rows")

    return lab_assigned[
        ["icustay_id", "charttime", "signal_name", "valuenum", "source_table"]
    ].copy()


def apply_phys_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with valuenum outside physiological plausibility bounds."""
    total_before = len(df)
    drop_counts: dict[str, int] = {}

    rows_to_keep = pd.Series(True, index=df.index)
    for sig, (lo, hi) in PHYS_BOUNDS.items():
        sig_mask = df.signal_name == sig
        oob      = sig_mask & ((df.valuenum < lo) | (df.valuenum > hi))
        n_drop   = oob.sum()
        if n_drop:
            drop_counts[sig] = n_drop
            rows_to_keep &= ~oob

    df = df[rows_to_keep].copy()
    total_dropped = total_before - len(df)

    if total_dropped:
        print(f"  Physiological range filter: dropped {total_dropped:,} rows")
        for sig, n in sorted(drop_counts.items()):
            lo, hi = PHYS_BOUNDS[sig]
            print(f"    {sig:15s}: {n:,} rows outside [{lo}, {hi}]")
    else:
        print("  Physiological range filter: 0 rows dropped (all values in range)")
    return df


# ── Split & write ─────────────────────────────────────────────────────────────

def split_and_write(df: pd.DataFrame, out_dir: pathlib.Path,
                    n_files: int) -> list[dict]:
    """
    Write df (sorted by charttime) into n_files CSV files.
    Returns manifest entries list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    chunk_size = max(1, n // n_files)
    # last file absorbs remainder
    boundaries = list(range(0, n, chunk_size))
    if len(boundaries) > n_files:
        boundaries = boundaries[:n_files]

    manifest_entries = []
    digits = len(str(n_files))

    for i, start in enumerate(boundaries, 1):
        end   = boundaries[i] if i < len(boundaries) else n
        chunk = df.iloc[start:end]
        fname = f"part-{str(i).zfill(5)}.csv"
        fpath = out_dir / fname
        chunk.to_csv(fpath, index=False)
        manifest_entries.append({
            "filename":     fname,
            "row_count":    len(chunk),
            "min_charttime": str(chunk.charttime.min()),
            "max_charttime": str(chunk.charttime.max()),
        })
        if i % 10 == 0 or i == len(boundaries):
            print(f"  Written {i}/{len(boundaries)} files…")

    return manifest_entries


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, manifest: list[dict],
                  out_dir: pathlib.Path) -> None:
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total rows          : {len(df):,}")
    print(f"  Unique icustay_ids  : {df.icustay_id.nunique()}")
    print(f"  Time span           : {df.charttime.min()}  ->  {df.charttime.max()}")
    print(f"  Output files        : {len(manifest)}")
    print(f"  Rows per file       : ~{len(df)//len(manifest):,}")
    print()

    print("  Rows per signal:")
    sig_counts = df.groupby("signal_name").size().sort_values(ascending=False)
    for sig, cnt in sig_counts.items():
        src_tags = df[df.signal_name == sig].source_table.value_counts().to_dict()
        src_str  = " | ".join(f"{k}:{v:,}" for k, v in src_tags.items())
        print(f"    {sig:15s}: {cnt:>7,}   ({src_str})")

    print()
    rows_per_stay = df.groupby("icustay_id").size()
    print("  Rows per icustay_id:")
    print(f"    mean   : {rows_per_stay.mean():,.1f}")
    print(f"    median : {rows_per_stay.median():,.1f}")
    print(f"    min    : {rows_per_stay.min()}")
    print(f"    max    : {rows_per_stay.max()}")
    print()

    # Histogram (10 buckets)
    counts_arr = rows_per_stay.values
    buckets, edges = np.histogram(counts_arr, bins=10)
    print("  Histogram of rows per stay:")
    for lo, hi, cnt in zip(edges[:-1], edges[1:], buckets):
        bar = "#" * min(cnt, 40)
        print(f"    [{int(lo):5d}–{int(hi):5d}]: {cnt:3d} stays  {bar}")
    print()

    # Manifest file
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "created":    datetime.utcnow().isoformat() + "Z",
            "total_rows": len(df),
            "n_files":    len(manifest),
            "signals":    sorted(df.signal_name.unique().tolist()),
            "time_span": {
                "min": str(df.charttime.min()),
                "max": str(df.charttime.max()),
            },
            "files": manifest,
        }, f, indent=2)
    print(f"  Manifest written: {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build chunked CSV stream-input for Spark Structured Streaming")
    ap.add_argument("--data-dir", default="data/raw",       type=pathlib.Path)
    ap.add_argument("--cohort",   default="data/processed/cohort.parquet",
                    type=pathlib.Path)
    ap.add_argument("--out-dir",  default="data/stream_input", type=pathlib.Path)
    ap.add_argument("--n-files",  default=50, type=int)
    args = ap.parse_args()

    data_dir: pathlib.Path = args.data_dir
    out_dir:  pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cohort ───────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1  Loading cohort")
    print("=" * 60)
    if not args.cohort.exists():
        print(f"ERROR: cohort file not found: {args.cohort}", file=sys.stderr)
        sys.exit(1)
    cohort = pd.read_parquet(args.cohort)
    icu    = pd.read_csv(data_dir / "ICUSTAYS.csv", low_memory=False)
    cohort_ids = set(cohort.icustay_id.astype(int).tolist())
    print(f"  Cohort ICU stays: {len(cohort_ids)}  "
          f"(septic: {cohort.sepsis_label.sum()}, "
          f"non-septic: {(cohort.sepsis_label==0).sum()})")

    # ── Build vitals from CHARTEVENTS ─────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2  Loading vitals from CHARTEVENTS")
    print("=" * 60)
    iid_map = _itemid_to_signal()
    vitals  = build_vitals(data_dir / "CHARTEVENTS.csv", cohort_ids, iid_map)
    n_vitals_raw = len(vitals)

    # ── Build labs from LABEVENTS ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3  Loading labs from LABEVENTS")
    print("=" * 60)
    lab_iid_map = _lab_itemid_to_signal()
    labs        = build_labs(data_dir / "LABEVENTS.csv", icu, cohort_ids, lab_iid_map)
    n_labs_raw  = len(labs)

    # ── Combine ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 4  Combining and applying physiological filters")
    print("=" * 60)
    combined = pd.concat([vitals, labs], ignore_index=True)
    print(f"  Combined rows (pre-filter): {len(combined):,}  "
          f"(vitals: {n_vitals_raw:,}, labs: {n_labs_raw:,})")

    combined = apply_phys_bounds(combined)

    # ── Sort globally by charttime ─────────────────────────────────────────
    print()
    print("  Sorting by charttime…")
    combined["charttime"] = pd.to_datetime(combined["charttime"])
    combined = combined.sort_values("charttime").reset_index(drop=True)
    # Coerce charttime back to string for CSV (consistent ISO format)
    combined["charttime"] = combined["charttime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Sort complete. Total rows: {len(combined):,}")

    # ── Split and write ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"STEP 5  Writing {args.n_files} CSV files to {out_dir}")
    print("=" * 60)
    manifest = split_and_write(combined, out_dir, args.n_files)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(combined, manifest, out_dir)
    print("Step 02 complete.")


if __name__ == "__main__":
    main()
