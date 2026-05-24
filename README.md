# Sepsis Early Warning — Streaming Pipeline

End-to-end pipeline for early sepsis prediction from MIMIC-III ICU time-series data.
Uses Apache Kafka + Spark Structured Streaming + XGBoost + LSTM ensemble.

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11.8 |
| Apache Spark | 4.1.1 |
| Docker Desktop | 4.x |
| Java | 11+ |

```powershell
# Activate virtualenv (from project root)
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install kafka-python pandas pyarrow pyspark xgboost torch scikit-learn optuna
```

---

## Pipeline Steps

### Step 01 — Cohort labeling

```powershell
$env:PYSPARK_PYTHON = ".\.venv\Scripts\python.exe"
python src/01_prepare_cohort.py
```

### Step 02 — Build stream input chunks

```powershell
python src/02_build_stream_input.py
```

Produces 50 CSV chunks in `data/stream_input/` (107,768 rows total across 136 patients).

---

## Step 03 — Kafka Streaming (RECOMMENDED)

### 3a. Start Kafka

```powershell
docker compose -f docker/docker-compose.yml up -d
```

Wait ~30 seconds for Kafka to become healthy, then verify:

```powershell
docker compose -f docker/docker-compose.yml ps
# Both zookeeper and kafka should show status "healthy"
```

### 3b. Run the producer

The producer replays all 50 CSV chunks through the `mimic-icu-stream` Kafka topic
at 5,000 messages/sec (~22 seconds total) and saves timing to
`data/processed/producer_timing.json`.

```powershell
$env:PYSPARK_PYTHON = ".\.venv\Scripts\python.exe"
python src/03a_kafka_producer.py
# Options: --bootstrap localhost:9092  --rate 5000
```

### 3c. Run the Spark streaming job (Kafka source)

Clear any previous output first, then run:

```powershell
# Clear previous run's output (required if comparing Kafka vs CSV row counts)
Remove-Item -Recurse -Force data\stream_output -ErrorAction SilentlyContinue

# Option A — via spark-submit (Kafka JAR downloaded automatically)
$env:PYSPARK_PYTHON = ".\.venv\Scripts\python.exe"
spark-submit.cmd `
    --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 `
    src/03_spark_streaming_job.py --source kafka

# Option B — via Python (same JAR is downloaded automatically)
python src/03_spark_streaming_job.py --source kafka
```

Expected output: **~28,890 windows** across 136 patients (same as CSV run).

### 3d. Measure streaming latency (Task 2)

After the Spark job finishes:

```powershell
python src/03c_measure_latency.py
# Saves data/processed/latency_results.json with p50/p95/p99 latency and throughput
```

---

## Step 03 — CSV Fallback (original behaviour)

```powershell
python src/03_spark_streaming_job.py --source csv
```

---

## Step 03b — Prepare training data

```powershell
$env:PYSPARK_PYTHON = ".\.venv\Scripts\python.exe"
python src/03b_prepare_training_data.py
```

## Step 04 — XGBoost training

```powershell
python src/04_train_xgboost.py
# With Optuna tuning (~5 min): python src/04_train_xgboost.py --tune
```

Saves `data/processed/training_time_xgboost.json` with per-fold and total time.

## Step 05 — LSTM training

```powershell
python src/05_train_lstm.py
```

Saves `data/processed/training_time_lstm.json` with per-fold and total time.

## Step 06 — Evaluation & ensemble

```powershell
python src/06_evaluate.py
```

---

## Stopping Kafka

```powershell
# Stop containers (keep data volumes)
docker compose -f docker/docker-compose.yml down

# Full reset (wipe Kafka topic data)
docker compose -f docker/docker-compose.yml down -v
```

---

## Key output files

| File | Description |
|------|-------------|
| `data/stream_output/*.parquet` | 28,890 windowed feature rows |
| `data/processed/latency_log.csv` | Per-window latency stats (Kafka path) |
| `data/processed/latency_results.json` | p50/p95/p99 latency + throughput |
| `data/processed/producer_timing.json` | Producer wall-clock timing |
| `data/processed/training_time_xgboost.json` | XGBoost per-fold training time |
| `data/processed/training_time_lstm.json` | LSTM per-fold training time |
| `data/processed/oof_xgb.parquet` | XGBoost OOF predictions |
| `data/processed/oof_lstm.parquet` | LSTM OOF predictions |
| `data/processed/oof_ensemble.parquet` | Ensemble OOF predictions |
| `docs/hardware_specs.txt` | CPU / RAM / OS for paper |
| `docs/paper_numbers.md` | All paper-ready metrics in one place |

---

## Troubleshooting

**`spark.jars.packages` download fails**  
Ensure internet access and Maven Central is reachable. Check `~/.ivy2/` for cached JARs.
If version `4.1.1` is unavailable, try `4.0.0`:
```
--packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0
```

**Kafka connection refused**  
Wait longer after `docker compose up -d` (Kafka takes 20–30 s to initialize).
Verify with: `docker compose -f docker/docker-compose.yml ps`

**Docker not available**  
Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
or use a local Confluent Kafka installation (Confluent CLI).
