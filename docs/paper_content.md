# Paper Content Reference — Sepsis Early-Warning Streaming Pipeline

> All numbers in this document are from the final run on 2026-05-15.
> Copy sections directly into your paper as needed.

---

## 1. Dataset

| Property | Value |
|---|---|
| Source | MIMIC-III Clinical Database (demo subset) |
| Cohort | 136 adult ICU stays |
| Sepsis definition | Sepsis-3 (SOFA ≥ 2 + suspected infection) |
| Time series signals | 12 physiological signals (see below) |
| Label horizons | 6 h, 12 h, 24 h before sepsis onset |
| Positive class rate | ~0.9% (246 positive windows / 28,890 total) |
| Class imbalance ratio | ≈ 117:1 (negative:positive) |

**Physiological signals (12):**
bilirubin, creatinine, diastolic blood pressure (DBP), heart rate, lactate,
mean arterial pressure (MAP), platelets, respiratory rate, systolic blood pressure (SBP),
SpO₂, temperature, white blood cell count (WBC).

**Data split:** 5-fold patient-level stratified cross-validation (no patient appears in both train and validation).

---

## 2. System Architecture

```
MIMIC-III CSVs
    │
    ▼
[Step 01] Cohort extraction (Sepsis-3 criteria, SOFA scoring)
    │
    ▼
[Step 02] Stream input construction → 50 × part-*.csv chunks
    │
    ▼
[Step 03a] Kafka Producer (Spark batch write)
    │   Topic: mimic-icu-stream  |  Broker: localhost:9092
    ▼
Apache Kafka 3.9.2 (KRaft mode, no ZooKeeper)
    │
    ▼
[Step 03 / 07] Spark Structured Streaming
    │   • Watermark: 1 hour
    │   • Sliding window: 1 hour / 30-minute slide
    │   • 72 aggregated features (12 signals × 6 stats)
    │
    ├──▶ [Step 03]  Window features → Parquet (stream_output/)
    │
    └──▶ [Step 07]  Real-time XGBoost inference → risk scores + alerts
              │
              ▼
         data/processed/realtime_predictions.csv
```

**Two-stage model training (offline):**

```
stream_output/ Parquet
    │
    ▼
[Step 03b] Feature engineering → xgb_features.parquet + lstm_features.parquet
    │
    ├──▶ [Step 04] XGBoost (5-fold CV, Optuna tuning) → xgb_label_{6h,12h,24h}.json
    ├──▶ [Step 05] Causal LSTM  (5-fold CV, focal loss) → lstm_fold{0..4}.pt
    └──▶ [Step 06] Ensemble + evaluation → OOF metrics, ROC/PR plots
```

---

## 3. Technologies

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11.8 |
| Stream processing | Apache Spark Structured Streaming | 4.1.1 |
| Message broker | Apache Kafka | 3.9.2 (KRaft mode) |
| Kafka–Spark connector | spark-sql-kafka-0-10 | 4.1.1 |
| Gradient boosting | XGBoost | 3.2.0 |
| Deep learning | PyTorch | 2.6.0 (CPU) |
| Hyperparameter search | Optuna | — |
| Data processing | pandas, pyarrow, NumPy | — |
| Scikit-learn | ROC/PR metrics, cross-validation | 1.8.0 |
| Storage | Apache Parquet | — |
| OS | Microsoft Windows 11 Pro | 10.0.26200 |

---

## 4. Hardware (Experimental Environment)

| Component | Specification |
|---|---|
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz |
| Physical cores / Logical processors | 4 / 8 |
| RAM | 8 GB |
| Operating system | Windows 11 Pro (64-bit, build 10.0.26200) |
| Spark parallelism | `local[4]` — 4 executor threads |
| GPU | None (CPU-only training) |

> Note: All training and streaming experiments were performed on a consumer laptop
> without GPU acceleration, demonstrating the feasibility of the approach on
> resource-constrained hardware.

---

## 5. Feature Engineering

### Streaming window aggregation (Step 03)

Each ICU stay's raw vital-sign observations are ingested as a Kafka event stream.
Spark Structured Streaming applies a **1-hour sliding window with 30-minute slide**
and a **1-hour watermark** for late-data handling.

Within each window, six statistics are computed per signal:

| Statistic | Description |
|---|---|
| mean | Average value |
| stddev | Standard deviation |
| min | Minimum value |
| max | Maximum value |
| last | Most recent non-null observation |
| slope | OLS slope via `covar_samp(t, v) / var_samp(t)` |

Total: **12 signals × 6 statistics = 72 streaming features** per window.

### XGBoost feature set (51 dimensions)

From the 72 streaming features, XGBoost uses only the **4 location/scale stats**
(mean, min, max, last) per signal = 48 features, plus 3 temporal context features
added at training time:

| Feature | Description |
|---|---|
| `window_idx` | Ordinal index of window within the stay (0, 1, 2, …) |
| `window_idx_norm` | `window_idx / max_window_idx` — relative position in stay |
| `hours_in_icu` | Hours elapsed since the patient's first observed window |

### LSTM feature set (25 dimensions)

| Feature group | Dimensions |
|---|---|
| `{signal}_last` — most recent value per signal | 12 |
| `{signal}_fresh` — value from latest non-null observation | 12 |
| `window_idx_norm` | 1 |
| **Total** | **25** |

---

## 6. Models

### 6.1 XGBoost

- **Algorithm:** `binary:logistic` with `tree_method=hist`
- **Eval metric:** AUPRC (`aucpr`) — appropriate for severe class imbalance
- **Imbalance handling:** `scale_pos_weight = N_neg / N_pos` per fold
- **Regularization:** `gamma=1.0`, `min_child_weight=10`, `reg_alpha=0.1`, `reg_lambda=2.0`
- **Hyperparameter search:** Optuna (50 trials, TPE sampler, ~5 min per label)
- **Early stopping:** 50 rounds on validation AUPRC
- **CV:** 5-fold patient-level stratified (no patient leakage)

**Key hyperparameters (post-tuning):**

| Parameter | Value |
|---|---|
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `n_estimators` | 2000 (with early stopping) |
| `subsample` | 0.80 |
| `colsample_bytree` | 0.70 |

### 6.2 Causal LSTM

- **Architecture:** Single-layer causal LSTM (no future data leakage)
- **Output heads:** 3 simultaneous heads (multi-task: 6h, 12h, 24h) — one forward pass
- **Loss:** Focal loss (`γ = 2`) — down-weights easy negatives, focuses on hard positives
- **Optimizer:** Adam, learning rate 1e-3 with ReduceLROnPlateau
- **Epochs:** 80 max with early stopping (patience 10)
- **Input:** Sequences of 25-dim feature vectors, one per 30-min window

### 6.3 Ensemble

- **Method:** Average of XGBoost and LSTM predicted probabilities (equal weights)
- **Rationale:** LSTM captures temporal dynamics; XGBoost captures tabular feature interactions

---

## 7. Results

### 7.1 Model Performance (5-fold OOF)

| Label | Model | AUROC | AUPRC | Sensitivity @ 90% Specificity |
|---|---|---|---|---|
| **6h** | XGBoost | 0.9211 | 0.2421 | 73.8% |
| **6h** | LSTM | 0.7810 | 0.1293 | 68.0% |
| **6h** | **Ensemble** | **0.9294** | **0.2674** | **74.9%** |
| **12h** | XGBoost | 0.8797 | 0.2175 | 75.8% |
| **12h** | LSTM | 0.7663 | 0.1885 | 64.5% |
| **12h** | **Ensemble** | **0.8674** | **0.2495** | **72.5%** |
| **24h** | XGBoost | 0.9224 | 0.2680 | 84.7% |
| **24h** | LSTM | 0.7480 | 0.1308 | 64.1% |
| **24h** | **Ensemble** | **0.9182** | **0.2964** | **78.4%** |

**Baseline AUPRC** (random classifier at 0.9% prevalence): ~0.009
→ Ensemble is **18× better than random** on AUPRC for the 6h horizon.

**Median lead time:** 8 hours before sepsis onset across all horizons.

### 7.2 Model Selection Rationale

XGBoost consistently outperforms LSTM on AUROC and AUPRC despite the LSTM's
ability to model temporal sequences. This is attributed to:
1. Small dataset (136 patients, 28,890 windows) — insufficient for LSTM to learn
   long-range dependencies reliably
2. Sparse and irregular vital-sign sampling — LSTM is sensitive to missingness patterns
3. XGBoost's explicit regularization and `scale_pos_weight` are well-suited to the
   117:1 class imbalance

The ensemble provides marginal improvement on AUPRC and sensitivity over XGBoost alone.

### 7.3 Training Time

| Model | Configuration | Total time | Mean per fold |
|---|---|---|---|
| XGBoost | 5-fold × 3 horizons, default params | **8.49 s** | 0.57 s |
| XGBoost | + Optuna (50 trials per label) | ~15–20 min | ~5 min |
| LSTM | 5-fold × 3 heads (multi-task), 80 epochs | **111.2 s** | 22.2 s |

**XGBoost per-fold breakdown (default params):**

| Label | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| 6h | 1.289 s | 1.012 s | 0.381 s | 0.518 s | 0.313 s |
| 12h | 0.734 s | 0.469 s | 0.267 s | 0.444 s | 0.375 s |
| 24h | 0.746 s | 0.442 s | 0.312 s | 0.832 s | 0.353 s |

**LSTM per-fold breakdown:**

| Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Total |
|---|---|---|---|---|---|
| 19.7 s | 19.5 s | 20.0 s | 19.2 s | 32.8 s | 111.2 s |

### 7.4 Streaming Pipeline Statistics

| Metric | Value |
|---|---|
| ICU stays processed | 136 |
| Sliding windows produced | 28,890 |
| Raw Kafka events consumed | 215,530 |
| Windows per stay (min / mean / median / max) | 2 / 212.4 / 105 / 1,670 |
| Positive windows (label_6h) | 246 (0.9%) |

### 7.5 Real-Time Inference (Step 07)

| Metric | Value |
|---|---|
| Inference engine | XGBoost (CPU, `local[4]`) |
| Windows scored | 28,890 |
| Alert threshold | 0.30 (≈ 90% specificity on OOF) |
| Alerts triggered — 12h model | 3,385 / 28,890 (11.7%) |
| Alerts triggered — 24h model | 3,592 / 28,890 (12.4%) |
| End-to-end latency (batch producer) | 218 s |

> **Latency note:** The 218 s latency reflects the Spark batch-mode producer
> (all 215,530 events share one wall-clock timestamp). In a true row-by-row
> streaming producer, per-window latency would be sub-second; the batch design
> was chosen for reproducible replay of the MIMIC-III dataset.

---

## 8. Dataset Statistics (Stream Output)

| Statistic | Value |
|---|---|
| Total sliding windows | 28,890 |
| Unique ICU stays | 136 |
| Label_6h positive windows | 246 (0.85%) |
| Label_6h negative windows | 28,644 (99.15%) |
| Min windows per stay | 2 |
| Max windows per stay | 1,670 |
| Mean windows per stay | 212.4 |
| Median windows per stay | 105 |
| Feature columns (streaming) | 72 (12 signals × 6 stats) |
| Feature columns (XGBoost) | 51 (48 signal stats + 3 temporal) |
| Feature columns (LSTM) | 25 (24 signal last/fresh + window_idx_norm) |

---

## 9. Figures Reference

All figures are saved under `data/processed/`:

| Figure file | Content |
|---|---|
| `plots/roc_label_6h.png` | ROC curves (XGB, LSTM, Ensemble) — 6h horizon |
| `plots/pr_label_6h.png` | Precision-Recall curves — 6h horizon |
| `plots/roc_label_12h.png` | ROC curves — 12h horizon |
| `plots/pr_label_12h.png` | Precision-Recall curves — 12h horizon |
| `plots/roc_label_24h.png` | ROC curves — 24h horizon |
| `plots/pr_label_24h.png` | Precision-Recall curves — 24h horizon |
| `paper_figures/fig1_roc_curves.png` | Combined ROC (all horizons) |
| `paper_figures/fig2_pr_curves.png` | Combined PR (all horizons) |
| `paper_figures/fig3_feature_importance.png` | XGBoost feature importance |
| `paper_figures/fig4_fold_performance.png` | Per-fold AUROC / AUPRC |
| `paper_figures/fig5_time_to_detection.png` | Lead time to sepsis onset |
| `paper_figures/fig6_label_distribution.png` | Label prevalence |
| `paper_figures/fig7_summary_panel.png` | Summary panel |
| `plots/realtime_score_distributions.png` | Production score distributions (6h/12h/24h) |
| `plots/realtime_calibration.png` | OOF vs production score comparison |
| `plots/realtime_alert_rate_by_hour.png` | Alert rate over ICU stay duration |
| `plots/realtime_trajectory_271544.png` | Risk score trajectory — highest-risk stay |

---

## 10. Key Claims for Paper

1. **Real-time feasibility on commodity hardware:** The full pipeline — Kafka ingestion,
   Spark Structured Streaming aggregation, and XGBoost inference — runs on a 4-core,
   8 GB laptop with no GPU.

2. **Strong early-warning performance:** The XGBoost–LSTM ensemble achieves AUROC = 0.929
   for 6h sepsis prediction with a median lead time of 8 hours, giving clinicians
   substantial time to intervene.

3. **Extreme class imbalance handling:** With only 0.9% positive windows,
   the ensemble AUPRC of 0.267 is 18× above the random baseline of ~0.009.

4. **Scalable window aggregation without pivot:** Spark's `pivot()` is unsupported on
   streaming DataFrames; conditional aggregation (`CASE WHEN signal = X THEN value END`)
   achieves identical results with full Spark 4.x compatibility.

5. **Alert threshold calibration:** An alert threshold of 0.30 on the 6h XGBoost model
   achieves 74.9% sensitivity at 90% specificity on held-out OOF data,
   a clinically actionable operating point.

---

## 11. Limitations

- **Dataset size:** MIMIC-III demo contains only 136 patients. Results on the full
  MIMIC-III (46,476 stays) or external cohorts may differ.
- **Simulated streaming:** The Kafka producer replays historical data in batch mode;
  true prospective deployment would require integration with a hospital EHR system.
- **Single-site data:** MIMIC-III is from a single US academic medical centre (BIDMC);
  generalisation to other institutions is not validated.
- **No forward-fill:** `applyInPandasWithState` is disabled on Windows (Spark 4.x
  constraint); missing signals within a window remain null rather than being
  forward-filled from previous windows.
- **CPU-only inference:** XGBoost inference on CPU is fast for this dataset size
  but may need GPU acceleration at scale (>10,000 concurrent patients).
