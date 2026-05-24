"""
01_prepare_cohort.py  —  Sepsis-3 labeling on MIMIC-III demo dataset.

Follows the MIT-LCP mimic-code Sepsis-3 reference implementation:
  https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts/sepsis

Suspicion of Infection (SOI):
  Antibiotic-THEN-culture: culture must occur within 24 h AFTER antibiotic.
    → suspected_infection_time = antibiotic_time
  Culture-THEN-antibiotic: antibiotic must occur within 72 h AFTER culture.
    → suspected_infection_time = culture_time
  Per stay: use the earliest valid suspected_infection_time (t_soi).
  Antibiotics restricted to systemic routes (IV, PO, PO/NG, NG, PB).

Sepsis onset (anchored on t_soi per stay):
  baseline_sofa = min SOFA in [max(icu_intime, t_soi − 48 h), t_soi].
  Onset = first hour in [t_soi − 48 h, t_soi + 24 h] where
          SOFA − baseline_sofa ≥ 2.

SOFA components (6-component standard):
  Respiratory  — PaO2/FiO2 (labs + chartevents); SpO2-based proxy if unavailable
  Coagulation  — platelet count (LABEVENTS 51265)
  Liver        — total bilirubin (LABEVENTS 50885)
  Cardiovascular — MAP (CHARTEVENTS) + vasopressor flag (INPUTEVENTS)
  CNS          — GCS total (198 CareVue; sum 220739+223900+223901 Metavision)
  Renal        — serum creatinine (LABEVENTS 50912)

Usage:
    python src/01_prepare_cohort.py
    python src/01_prepare_cohort.py --data-dir data/raw --out-dir data/processed
"""

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Item-ID tables ────────────────────────────────────────────────────────────

VITALS_ITEMIDS = {
    "hr":   [211, 220045],
    "sbp":  [51, 442, 455, 6701, 220050, 220179, 225309],
    "dbp":  [8368, 8440, 8441, 220051, 220180, 225310],
    "map":  [52, 456, 220052, 220181],
    "rr":   [615, 618, 220210, 224690],
    "temp": [676, 677, 678, 679, 223761, 223762],
    "spo2": [646, 220277],
}

# Fahrenheit itemids — values need °C conversion
TEMP_F_ITEMIDS = {678, 679, 223761}

FIO2_ITEMIDS = [189, 190, 191, 3420, 3422, 2981, 223835, 227009, 227010]

GCS_CV_ITEMID   = 198
GCS_MV_ITEMIDS  = [220739, 223900, 223901]   # Eye + Verbal + Motor

SOFA_LAB_ITEMIDS = {
    "pao2":       [50821],
    "platelets":  [51265],
    "bilirubin":  [50885],
    "creatinine": [50912],
}

VASO_MV_ITEMIDS = [221906, 221289, 221662, 221653, 222315, 221749]  # norepi,epi,dopa,dobu,vaso,phenyl
VASO_CV_ITEMIDS = [30120, 30047, 30044, 30119, 30309, 30043, 30307, 30042, 30306, 30051]

ANTIBIOTIC_KEYWORDS = [
    "vancomycin", "piperacillin", "meropenem", "cefazolin", "ceftriaxone",
    "ciprofloxacin", "levofloxacin", "metronidazole", "ampicillin", "gentamicin",
    "tobramycin", "amoxicillin", "azithromycin", "clindamycin", "linezolid",
    "daptomycin", "cefepime", "imipenem", "trimethoprim", "sulfameth",
    "fluconazole", "micafungin", "caspofungin", "nafcillin", "oxacillin",
    "penicillin", "erythromycin", "tetracycline", "doxycycline", "rifampin",
    "colistin", "polymyxin", "cephalexin", "cefpodoxime",
]

# Systemic routes only — exclude topical (RIGHT EYE, LEFT EYE, OU, TOP, PR)
SYSTEMIC_ROUTES = {"IV", "IV DRIP", "IVPB", "PO", "PO/NG", "NG", "PB", "SL", "ORAL"}

# ── SOFA scoring functions ────────────────────────────────────────────────────

def _spo2_to_pf(spo2: float) -> float:
    """Rough SpO2 → PaO2/FiO2 proxy (assumes no supplemental O2)."""
    if pd.isna(spo2):
        return np.nan
    if spo2 >= 97: return 400.0
    if spo2 >= 95: return 300.0
    if spo2 >= 90: return 200.0
    if spo2 >= 80: return 100.0
    return 50.0


def _score_resp(pf: float) -> float:
    if pd.isna(pf): return np.nan
    if pf > 400: return 0
    if pf > 300: return 1
    if pf > 200: return 2
    if pf > 100: return 3
    return 4


def _score_platelets(val: float) -> float:
    if pd.isna(val): return np.nan
    if val > 150: return 0
    if val > 100: return 1
    if val > 50:  return 2
    if val > 20:  return 3
    return 4


def _score_bilirubin(val: float) -> float:
    if pd.isna(val): return np.nan
    if val < 1.2:  return 0
    if val < 2.0:  return 1
    if val < 6.0:  return 2
    if val < 12.0: return 3
    return 4


def _score_creatinine(val: float) -> float:
    if pd.isna(val): return np.nan
    if val < 1.2: return 0
    if val < 2.0: return 1
    if val < 3.5: return 2
    if val < 5.0: return 3
    return 4


def _score_gcs(val: float) -> float:
    if pd.isna(val): return np.nan
    if val >= 15: return 0
    if val >= 13: return 1
    if val >= 10: return 2
    if val >= 6:  return 3
    return 4


def _score_map(map_val: float, on_vaso: bool) -> float:
    if on_vaso:
        return 2  # minimum for any vasopressor
    if pd.isna(map_val): return np.nan
    return 0 if map_val >= 70 else 1


def _sofa_total(components: list) -> float:
    """Sum non-NaN components. If all NaN → 0 (conservative / unknown)."""
    valid = [c for c in components if not pd.isna(c)]
    return float(sum(valid))


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _load(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        print(f"  ERROR: required file missing — {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False, **kwargs)
    if df.empty:
        print(f"  WARNING: {path.name} loaded but is empty")
    return df


def _dtcols(df: pd.DataFrame, *cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# ── SOI detection ─────────────────────────────────────────────────────────────

def find_soi_times(rx: pd.DataFrame, micro: pd.DataFrame,
                   icu: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per stay: icustay_id, suspected_infection_time.

    Pairing rules (mimic-code reference):
      abx → culture within 24 h: SOI = abx_time
      culture → abx within 72 h: SOI = culture_time
    Per stay, keep only the earliest valid SOI.
    Antibiotics restricted to systemic routes (SYSTEMIC_ROUTES).
    """
    # ── antibiotics ──────────────────────────────────────────────────────
    abx = rx[rx.drug.str.lower().str.contains(
        "|".join(ANTIBIOTIC_KEYWORDS), na=False
    )].copy()
    # route filter: exclude topical/rectal/ophthalmic
    if "route" in abx.columns:
        route_upper = abx.route.str.upper().str.strip()
        abx = abx[route_upper.isin(SYSTEMIC_ROUTES) | route_upper.isna()]
    abx = _dtcols(abx, "startdate")
    abx = abx.dropna(subset=["startdate", "hadm_id"])
    print(f"  Antibiotic Rx (systemic routes): {len(abx):,} rows, "
          f"{abx.hadm_id.nunique()} unique admissions")
    if "route" in abx.columns:
        print(f"  Route breakdown: {abx.route.value_counts().to_dict()}")

    # ── blood cultures ────────────────────────────────────────────────────
    blood = micro[micro.spec_type_desc.str.upper().str.contains(
        "BLOOD", na=False
    )].copy()
    blood = _dtcols(blood, "charttime", "chartdate")
    nat_mask = blood.charttime.isna()
    blood.loc[nat_mask, "charttime"] = blood.loc[nat_mask, "chartdate"]
    blood = blood.dropna(subset=["charttime", "hadm_id"])
    # deduplicate: one culture event per (hadm_id, charttime)
    blood = blood.drop_duplicates(subset=["hadm_id", "charttime"])
    print(f"  Blood cultures (deduped): {len(blood):,} rows, "
          f"{blood.hadm_id.nunique()} unique admissions")

    # ── map hadm_id → icustay_id ──────────────────────────────────────────
    hadm2icu = (icu.sort_values("intime")
                   .drop_duplicates("hadm_id", keep="first")
                   [["hadm_id", "icustay_id"]]
                   .rename(columns={"icustay_id": "icustay_id_icu"}))

    # PRESCRIPTIONS already carries icustay_id — drop it to avoid collision
    if "icustay_id" in abx.columns:
        abx = abx.drop(columns=["icustay_id"])
    abx = abx.merge(hadm2icu, on="hadm_id", how="inner").rename(
        columns={"icustay_id_icu": "icustay_id"})

    blood = blood.merge(hadm2icu, on="hadm_id", how="inner").rename(
        columns={"icustay_id_icu": "icustay_id"})

    # ── funnel counts ─────────────────────────────────────────────────────
    stays_abx  = set(abx.icustay_id.unique())
    stays_cult = set(blood.icustay_id.unique())
    stays_both = stays_abx & stays_cult
    print(f"  Stays with systemic abx : {len(stays_abx)}")
    print(f"  Stays with blood culture: {len(stays_cult)}")
    print(f"  Stays with both         : {len(stays_both)}")

    # ── pair per stay (mimic-code directional windows) ────────────────────
    soi_rows = []
    for sid in stays_both:
        a_times = abx[abx.icustay_id == sid].startdate.dropna().sort_values().values
        c_times = blood[blood.icustay_id == sid].charttime.dropna().sort_values().values

        best_soi: pd.Timestamp | None = None

        for t_abx in a_times:
            t_abx = pd.Timestamp(t_abx)
            for t_cult in c_times:
                t_cult = pd.Timestamp(t_cult)
                if t_abx <= t_cult:
                    # antibiotic first → culture within 24 h after
                    if (t_cult - t_abx) <= pd.Timedelta("24h"):
                        soi_t = t_abx
                        if best_soi is None or soi_t < best_soi:
                            best_soi = soi_t
                else:
                    # culture first → antibiotic within 72 h after
                    if (t_abx - t_cult) <= pd.Timedelta("72h"):
                        soi_t = t_cult
                        if best_soi is None or soi_t < best_soi:
                            best_soi = soi_t

        if best_soi is not None:
            soi_rows.append({
                "icustay_id":               sid,
                "suspected_infection_time": best_soi,
            })

    if not soi_rows:
        print("  WARNING: no valid SOI pairs found")
        return pd.DataFrame(columns=["icustay_id", "suspected_infection_time"])

    result = pd.DataFrame(soi_rows)
    print(f"  Valid SOI stays: {len(result)}")
    return result


# ── SOFA computation ──────────────────────────────────────────────────────────

def _ffill_lab_series(lab_sub: pd.DataFrame, hours: pd.DatetimeIndex,
                       t0: pd.Timestamp, item_ids: list) -> np.ndarray:
    """
    For lab values, return forward-filled hourly array.
    Takes last available value up to each hour (not just within the hour window).
    """
    sub = lab_sub[lab_sub.itemid.isin(item_ids)].set_index("charttime")["valuenum"].dropna()
    if sub.empty:
        return np.full(len(hours), np.nan)
    # resample to 1-h, forward-fill from first available value
    full_range = pd.date_range(t0, hours[-1], freq="1h")
    resampled = sub.resample("1h").last().reindex(full_range).ffill()
    return resampled.reindex(hours).values


def _last_in_window(ce_sub: pd.DataFrame, item_ids: list,
                    hours: pd.DatetimeIndex) -> np.ndarray:
    """Last chart value per 1-h bucket, then forward-fill across stay."""
    sub = ce_sub[ce_sub.itemid.isin(item_ids)].set_index("charttime")["valuenum"].dropna()
    if sub.empty:
        return np.full(len(hours), np.nan)
    resampled = sub.resample("1h").last().reindex(hours).ffill()
    return resampled.values


def _vaso_flags_mv(imv_vaso: pd.DataFrame, hours: pd.DatetimeIndex) -> np.ndarray:
    """True/False per hour if any vasopressor was running (Metavision)."""
    flags = np.zeros(len(hours), dtype=bool)
    for _, row in imv_vaso.iterrows():
        flags |= (hours >= row.starttime) & (hours <= row.endtime)
    return flags


def _vaso_flags_cv(icv_vaso: pd.DataFrame, hours: pd.DatetimeIndex) -> np.ndarray:
    """True/False per hour if a vasopressor charttime falls in the bucket."""
    if icv_vaso.empty:
        return np.zeros(len(hours), dtype=bool)
    flags = np.zeros(len(hours), dtype=bool)
    for t in icv_vaso.charttime:
        idx = hours.searchsorted(t, side="right") - 1
        if 0 <= idx < len(hours):
            flags[idx] = True
    return flags


def compute_hourly_sofa(icu: pd.DataFrame, ce: pd.DataFrame,
                         lab: pd.DataFrame, imv: pd.DataFrame,
                         icv: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame: icustay_id, hour, sofa (total), plus per-component columns.
    """
    icu = _dtcols(icu.copy(), "intime", "outtime")
    ce  = _dtcols(ce.copy(),  "charttime")
    lab = _dtcols(lab.copy(), "charttime")
    imv = _dtcols(imv.copy(), "starttime", "endtime")
    icv = _dtcols(icv.copy(), "charttime")

    # prefilter vasopressors
    imv_vaso = imv[imv.itemid.isin(VASO_MV_ITEMIDS)][["icustay_id", "starttime", "endtime"]].dropna()
    icv_vaso = icv[icv.itemid.isin(VASO_CV_ITEMIDS)][["icustay_id", "charttime"]].dropna()

    # fix temperature to Celsius in chartevents
    ce.loc[ce.itemid.isin(TEMP_F_ITEMIDS), "valuenum"] = \
        (ce.loc[ce.itemid.isin(TEMP_F_ITEMIDS), "valuenum"] - 32) * 5 / 9

    # join labs → icustay_id via hadm_id
    hadm2icu = (icu.sort_values("intime")
                   .drop_duplicates("hadm_id", keep="first")
                   [["hadm_id", "icustay_id"]])
    lab = lab.merge(hadm2icu, on="hadm_id", how="inner")

    # all chart item ids we need
    all_ce_ids = (
        [i for lst in VITALS_ITEMIDS.values() for i in lst]
        + [GCS_CV_ITEMID] + GCS_MV_ITEMIDS + FIO2_ITEMIDS
    )
    ce = ce[ce.itemid.isin(all_ce_ids)].dropna(subset=["valuenum", "icustay_id"])
    ce["icustay_id"] = ce["icustay_id"].astype(int)

    records = []
    n_stays = len(icu)

    for i, (_, stay) in enumerate(icu.iterrows(), 1):
        sid  = int(stay.icustay_id)
        t0, t1 = stay.intime, stay.outtime
        if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
            print(f"  [stay {sid}] skipped — invalid intime/outtime")
            continue

        hours = pd.date_range(t0.ceil("1h"), t1.floor("1h"), freq="1h")
        if len(hours) == 0:
            continue

        ce_s  = ce[ce.icustay_id == sid].sort_values("charttime")
        lab_s = lab[lab.icustay_id == sid].sort_values("charttime")
        imv_s = imv_vaso[imv_vaso.icustay_id == sid]
        icv_s = icv_vaso[icv_vaso.icustay_id == sid]

        # vitals (last value per 1-h bucket, then forward-fill)
        spo2_arr = _last_in_window(ce_s, VITALS_ITEMIDS["spo2"], hours)
        map_arr  = _last_in_window(ce_s, VITALS_ITEMIDS["map"],  hours)

        # FiO2 (/100 to get fraction)
        fio2_raw = _last_in_window(ce_s, FIO2_ITEMIDS, hours)
        with np.errstate(invalid="ignore"):
            fio2_arr = np.where(
                (fio2_raw > 0) & (fio2_raw <= 100), fio2_raw / 100.0, np.nan
            )

        # GCS: prefer CareVue total; fall back to Metavision sum
        gcs_cv_arr = _last_in_window(ce_s, [GCS_CV_ITEMID], hours)
        gcs_mv_sub = ce_s[ce_s.itemid.isin(GCS_MV_ITEMIDS)].copy()
        if not gcs_mv_sub.empty:
            gcs_mv_sum = gcs_mv_sub.groupby("charttime")["valuenum"].sum().reset_index()
            gcs_mv_sum = gcs_mv_sum.rename(columns={"valuenum": "gcs_mv"})
            gcs_mv_tmp = gcs_mv_sum.set_index("charttime")["gcs_mv"].resample("1h").last()
            gcs_mv_arr = gcs_mv_tmp.reindex(hours).ffill().values
        else:
            gcs_mv_arr = np.full(len(hours), np.nan)
        gcs_arr = np.where(~np.isnan(gcs_cv_arr), gcs_cv_arr, gcs_mv_arr)

        # labs (forward-fill from ICU admission)
        pao2_arr  = _ffill_lab_series(lab_s, hours, t0, [50821])
        plat_arr  = _ffill_lab_series(lab_s, hours, t0, [51265])
        bili_arr  = _ffill_lab_series(lab_s, hours, t0, [50885])
        creat_arr = _ffill_lab_series(lab_s, hours, t0, [50912])

        # PaO2/FiO2: use lab PaO2 + chart FiO2 if both available; else SpO2 proxy
        with np.errstate(invalid="ignore", divide="ignore"):
            pf_arr = np.where(
                (~np.isnan(pao2_arr)) & (~np.isnan(fio2_arr)) & (fio2_arr > 0),
                pao2_arr / fio2_arr,
                np.vectorize(_spo2_to_pf)(spo2_arr)
            )

        # vasopressor flag
        vaso_mv_flags = _vaso_flags_mv(imv_s, hours)
        vaso_cv_flags = _vaso_flags_cv(icv_s, hours)
        vaso_flags = vaso_mv_flags | vaso_cv_flags

        # score each component per hour (vectorized)
        s_resp   = np.vectorize(_score_resp)(pf_arr)
        s_coag   = np.vectorize(_score_platelets)(plat_arr)
        s_liver  = np.vectorize(_score_bilirubin)(bili_arr)
        s_cardio = np.array([_score_map(m, v) for m, v in zip(map_arr, vaso_flags)])
        s_cns    = np.vectorize(_score_gcs)(gcs_arr)
        s_renal  = np.vectorize(_score_creatinine)(creat_arr)

        # total SOFA: sum non-NaN components
        components = np.stack([s_resp, s_coag, s_liver, s_cardio, s_cns, s_renal], axis=1)
        sofa_total = np.nansum(components, axis=1)
        n_avail    = np.sum(~np.isnan(components), axis=1)

        for j, hr in enumerate(hours):
            records.append({
                "icustay_id": sid,
                "hour":       hr,
                "sofa":       float(sofa_total[j]),
                "n_components": int(n_avail[j]),
                "s_resp":    float(s_resp[j])    if not np.isnan(s_resp[j])    else None,
                "s_coag":    float(s_coag[j])    if not np.isnan(s_coag[j])    else None,
                "s_liver":   float(s_liver[j])   if not np.isnan(s_liver[j])   else None,
                "s_cardio":  float(s_cardio[j])  if not np.isnan(s_cardio[j])  else None,
                "s_cns":     float(s_cns[j])     if not np.isnan(s_cns[j])     else None,
                "s_renal":   float(s_renal[j])   if not np.isnan(s_renal[j])   else None,
            })

        if i % 20 == 0 or i == n_stays:
            print(f"  SOFA progress: {i}/{n_stays} stays, {len(records):,} hours so far")

    return pd.DataFrame(records)


# ── Sepsis onset ──────────────────────────────────────────────────────────────

def find_sepsis_onset(sofa_df: pd.DataFrame, soi_times_df: pd.DataFrame,
                      icu: pd.DataFrame) -> pd.DataFrame:
    """
    For each stay with a valid t_soi (suspected_infection_time):

    1. baseline_time = max(icu_intime, t_soi - 48 h)
       baseline_sofa = SOFA at baseline_time  (first available hourly slot
                       at or after baseline_time; prevents baseline=0 artefact
                       that arises when PRESCRIPTIONS date-precision places
                       t_soi before ICU intime).

    2. search_window = [t_soi - 48 h, t_soi + 24 h]
       onset = first hour in search_window where SOFA - baseline_sofa >= 2.

    Stays without SOI are labeled 0 / no_soi.
    """
    icu = _dtcols(icu.copy(), "intime", "outtime")
    icu_index = icu.set_index("icustay_id")[["intime"]].to_dict("index")
    all_sids  = icu.icustay_id.tolist()

    if soi_times_df.empty:
        soi_lookup: dict = {}
    else:
        soi_lookup = dict(zip(soi_times_df.icustay_id,
                              soi_times_df.suspected_infection_time))

    rows = []
    for sid in all_sids:
        stay_sofa = sofa_df[sofa_df.icustay_id == sid].sort_values("hour")

        if stay_sofa.empty:
            rows.append({"icustay_id": sid, "sepsis_label": 0,
                         "sepsis_onset_time": pd.NaT,
                         "excl_reason": "no_sofa_data"})
            continue

        if sid not in soi_lookup:
            rows.append({"icustay_id": sid, "sepsis_label": 0,
                         "sepsis_onset_time": pd.NaT,
                         "excl_reason": "no_soi"})
            continue

        t_soi  = pd.Timestamp(soi_lookup[sid])
        intime = pd.Timestamp(icu_index[sid]["intime"])

        hours_arr = stay_sofa.hour.values
        sofa_arr  = stay_sofa.sofa.values

        # ── baseline: SOFA at the reference point ────────────────────────
        # Using the first available SOFA slot at or after baseline_time
        # prevents the "baseline=0" inflation that occurs when
        # PRESCRIPTIONS.startdate (date-only) places t_soi before intime.
        baseline_time = max(intime, t_soi - pd.Timedelta("48h"))
        after_baseline = stay_sofa[stay_sofa.hour >= baseline_time]
        if not after_baseline.empty:
            baseline_sofa = float(after_baseline.iloc[0].sofa)
        else:
            # all data is before baseline_time; fall back to last SOFA
            baseline_sofa = float(stay_sofa.iloc[-1].sofa)

        # ── search window: [t_soi - 48 h, t_soi + 24 h] ─────────────────
        t_lo = t_soi - pd.Timedelta("48h")
        t_hi = t_soi + pd.Timedelta("24h")
        search_mask = (
            (stay_sofa.hour >= t_lo) & (stay_sofa.hour <= t_hi)
        ).values

        onset = None
        for idx in np.where(search_mask)[0]:
            if sofa_arr[idx] - baseline_sofa >= 2.0:
                onset = pd.Timestamp(hours_arr[idx])
                break

        if onset is not None:
            rows.append({"icustay_id": sid, "sepsis_label": 1,
                         "sepsis_onset_time": onset,
                         "excl_reason": "sepsis"})
        else:
            rows.append({"icustay_id": sid, "sepsis_label": 0,
                         "sepsis_onset_time": pd.NaT,
                         "excl_reason": "criteria_not_met"})

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Sepsis-3 cohort labeling — MIMIC-III demo")
    ap.add_argument("--data-dir", default="data/raw",       type=pathlib.Path)
    ap.add_argument("--out-dir",  default="data/processed", type=pathlib.Path)
    args = ap.parse_args()

    data_dir: pathlib.Path = args.data_dir
    out_dir:  pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tables ───────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1  Loading MIMIC-III tables")
    print("=" * 60)

    icu   = _load(data_dir / "ICUSTAYS.csv")
    rx    = _load(data_dir / "PRESCRIPTIONS.csv")
    micro = _load(data_dir / "MICROBIOLOGYEVENTS.csv")
    lab   = _load(data_dir / "LABEVENTS.csv")
    imv   = _load(data_dir / "INPUTEVENTS_MV.csv")
    icv   = _load(data_dir / "INPUTEVENTS_CV.csv")

    icu = _dtcols(icu, "intime", "outtime")

    print(f"  ICU stays  : {len(icu):,}  ({icu.subject_id.nunique()} patients)")
    print(f"  dbsource   : {icu.dbsource.value_counts().to_dict()}")

    # CHARTEVENTS is large — filter to only the itemids we need
    print("  Loading CHARTEVENTS (filtering on read)…")
    needed_ce_ids = set(
        [i for lst in VITALS_ITEMIDS.values() for i in lst]
        + [GCS_CV_ITEMID] + GCS_MV_ITEMIDS + FIO2_ITEMIDS
    )
    chunks = []
    for chunk in pd.read_csv(data_dir / "CHARTEVENTS.csv", chunksize=100_000,
                              low_memory=False):
        sub = chunk[chunk.itemid.isin(needed_ce_ids)]
        if len(sub):
            chunks.append(sub[["icustay_id", "itemid", "charttime", "valuenum"]])
    ce = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["icustay_id", "itemid", "charttime", "valuenum"])
    print(f"  CHARTEVENTS (filtered) : {len(ce):,} rows")

    # ── SOI times ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 2  Suspicion-of-Infection (SOI) times")
    print("=" * 60)
    soi_df = find_soi_times(rx, micro, icu)

    # ── Hourly SOFA ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 3  Hourly SOFA computation")
    print("=" * 60)
    sofa_df = compute_hourly_sofa(icu, ce, lab, imv, icv)
    n_hours = len(sofa_df)
    n_stays_sofa = sofa_df.icustay_id.nunique()
    print(f"  Hourly records : {n_hours:,}  across {n_stays_sofa} stays")
    print(f"  SOFA summary:")
    print(sofa_df.sofa.describe().round(2).to_string())

    # ── Sepsis onset ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("STEP 4  Sepsis-3 onset labeling")
    print("=" * 60)
    cohort = find_sepsis_onset(sofa_df, soi_df, icu)
    cohort = cohort.merge(
        icu[["icustay_id", "subject_id", "hadm_id", "intime", "outtime", "los", "dbsource"]],
        on="icustay_id", how="left",
    )

    # ── Cohort statistics ─────────────────────────────────────────────────
    total   = len(cohort)
    septic  = int(cohort.sepsis_label.sum())
    non_sep = total - septic
    prev    = septic / total if total > 0 else 0.0

    print()
    print("=" * 60)
    print("COHORT STATISTICS")
    print("=" * 60)
    print(f"  Total ICU stays      : {total}")
    print(f"  Septic stays         : {septic}")
    print(f"  Non-septic stays     : {non_sep}")
    print(f"  Sepsis prevalence    : {prev * 100:.1f}%")
    print()
    print("  Funnel / label breakdown:")
    for reason, cnt in cohort.excl_reason.value_counts().items():
        print(f"    {reason:28s}: {cnt}")

    if septic > 0:
        print()
        print("  Septic stay details:")
        disp_cols = ["icustay_id", "subject_id", "sepsis_onset_time", "intime", "los", "dbsource"]
        print(cohort.loc[cohort.sepsis_label == 1, disp_cols].to_string(index=False))
    else:
        print()
        print("  NOTE: 0 septic stays. Downstream models will use class-weighted training.")

    # ── Sanity check: prevalence must be in [5%, 40%] ─────────────────────
    PREV_LO, PREV_HI = 0.05, 0.40
    if not (PREV_LO <= prev <= PREV_HI):
        print()
        print("!" * 60)
        print(f"  SANITY CHECK FAILED: prevalence {prev*100:.1f}% outside [{PREV_LO*100:.0f}%, {PREV_HI*100:.0f}%]")
        print("  Funnel breakdown:")
        soi_n  = len(soi_df)
        sofa_n = sofa_df.icustay_id.nunique()
        print(f"    Total stays          : {total}")
        print(f"    Stays with SOI       : {soi_n}")
        print(f"    Stays with SOFA data : {sofa_n}")
        print(f"    Sepsis (dSOFA >= 2)  : {septic}")
        print(f"    Criteria not met     : {(cohort.excl_reason == 'criteria_not_met').sum()}")
        print(f"    No SOI               : {(cohort.excl_reason == 'no_soi').sum()}")
        print(f"    No SOFA data         : {(cohort.excl_reason == 'no_sofa_data').sum()}")
        print("!" * 60)
        sys.exit(1)

    # ── Save ─────────────────────────────────────────────────────────────
    cohort_path = out_dir / "cohort.parquet"
    sofa_path   = out_dir / "sofa_hourly.parquet"
    cohort.to_parquet(cohort_path, index=False)
    sofa_df.to_parquet(sofa_path, index=False)

    print()
    print(f"  Saved: {cohort_path}")
    print(f"  Saved: {sofa_path}  ({sofa_path.stat().st_size // 1024} KB)")
    print()
    print("Step 01 complete.")


if __name__ == "__main__":
    main()
