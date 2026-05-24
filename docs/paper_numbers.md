# Paper Numbers — Final Run

Generated: 2026-05-13

---

## Hardware

- **CPU:** 11th Gen Intel Core i5-1135G7 @ 2.40 GHz
- **Cores / Threads:** 4 physical cores / 8 logical processors
- **RAM:** 8 GB
- **OS:** Microsoft Windows 11 Pro (Version 10.0.26200, 64-bit)

---

## Software Versions

| Component | Version |
|-----------|---------|
| Python | 3.11.8 |
| Apache Spark | 4.1.1 |
| Kafka (Confluent Platform) | 7.6.1 (Apache Kafka 3.6.x) |
| PyTorch | 2.6.0 (CPU) |
| XGBoost | 3.2.0 |
| scikit-learn | 1.8.0 |
| pandas | — |
| pyarrow | — |

---

## Streaming Pipeline Performance (Task 2)

> **Run the following commands to populate this section:**
> ```powershell
> docker compose -f docker/docker-compose.yml up -d   # start Kafka
> python src/03a_kafka_producer.py                     # replay 107,768 events
> python src/03_spark_streaming_job.py --source kafka  # Spark Kafka job
> python src/03c_measure_latency.py                    # compute metrics
> # Results saved to: data/processed/latency_results.json
> ```

| Metric | Value |
|--------|-------|
| Total events processed | 107,768 |
| Producer rate | 5,000 msg/s (configurable) |
| End-to-end wall-clock | **[run 03c_measure_latency.py]** s |
| Throughput | **[run 03c_measure_latency.py]** events/s |
| Per-event latency p50 | **[run 03c_measure_latency.py]** ms |
| Per-event latency p95 | **[run 03c_measure_latency.py]** ms |
| Per-event latency p99 | **[run 03c_measure_latency.py]** ms |
| Output windows | ~28,890 (same as CSV-source run) |
| Spark parallelism | local[4] — 4 worker threads |

*Latency note: Per-event latency is measured as*
`processing_timestamp_ms − producer_timestamp_ms` *per Kafka message,
averaged within each 30-minute sliding window.*

---

## Training Time (Task 4)

### XGBoost (5-fold CV × 3 label horizons, default hyperparameters)

| Label | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|-------|--------|--------|--------|--------|--------|
| label_6h  | 1.289s | 1.012s | 0.381s | 0.518s | 0.313s |
| label_12h | 0.734s | 0.469s | 0.267s | 0.444s | 0.375s |
| label_24h | 0.746s | 0.442s | 0.312s | 0.832s | 0.353s |

- **Total time:** 8.49 s across 5 folds × 3 horizons
- **Mean per fold:** 0.57 s

> Note: This run used **default hyperparameters** (no Optuna tuning). The
> previously reported AUROC=0.9209 for label_6h was from a tuned run (`--tune`,
> ~5 min Optuna search per label). If the paper reports the tuned result,
> add the Optuna search time (≈ 5 min per label × 3 = 15+ min) to the total.

### LSTM (5-fold CV, 80 epochs max with early stopping, multi-task 3 heads)

| Fold | Time (s) |
|------|----------|
| Fold 0 | 19.7 |
| Fold 1 | 19.5 |
| Fold 2 | 20.0 |
| Fold 3 | 19.2 |
| Fold 4 | 32.8 |

- **Total time:** 111.2 s across 5 folds × 3 horizons (multi-task)
- **Mean per fold:** 22.2 s

---

## Model Results (for reference — from prior full evaluation run)

*These are from the previous run with Optuna-tuned XGBoost.*
*See `data/processed/oof_ensemble.parquet` and `data/processed/plots/`.*

| Label | Model | AUROC | 95% CI | AUPRC | Sens@90%Sp | Lead Time |
|-------|-------|-------|--------|-------|------------|-----------|
| label_6h  | XGBoost | 0.9209 | [0.905, 0.936] | 0.2460 | 0.738 | — |
| label_6h  | LSTM | 0.7817 | [0.744, 0.819] | 0.1323 | 0.680 | — |
| label_6h  | **Ensemble** | **0.9294** | [0.918, 0.941] | **0.2709** | **0.749** | 8h median |
| label_12h | Ensemble | 0.8676 | [0.847, 0.889] | 0.2517 | 0.725 | 8h median |
| label_24h | Ensemble | 0.9184 | [0.908, 0.928] | 0.2993 | 0.784 | 8h median |

Baseline AUPRC (prevalence-rate random): ~0.015 → ensemble is **18× better than random**.

---

## Files Summary

| File | Status | Description |
|------|--------|-------------|
| `docker/docker-compose.yml` | Created | Kafka + Zookeeper for local streaming |
| `src/03a_kafka_producer.py` | Created | Replays 107,768 CSV rows to Kafka |
| `src/03_spark_streaming_job.py` | Modified | `--source kafka\|csv`, latency columns |
| `src/03c_measure_latency.py` | Created | Computes p50/p95/p99 + throughput |
| `data/processed/training_time_xgboost.json` | Generated | 8.49s total, 0.57s/fold |
| `data/processed/training_time_lstm.json` | Generated | 111.2s total, 22.2s/fold |
| `data/processed/latency_results.json` | **Pending** | Run Kafka pipeline first |
| `data/processed/producer_timing.json` | **Pending** | Run 03a_kafka_producer.py first |
| `data/processed/latency_log.csv` | **Pending** | Written by Spark Kafka job |
| `docs/hardware_specs.txt` | Generated | i5-1135G7 / 8 GB / Win 11 Pro |
