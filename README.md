# Sepsis Early Warning — Real-Time Streaming Pipeline

This project builds a real-time pipeline to predict sepsis early using ICU data from MIMIC-III.
It streams patient vitals through Kafka, processes them with Spark Structured Streaming, and runs XGBoost + LSTM models to flag patients at risk.

## What I used
- Apache Kafka + Docker (for streaming)
- Apache Spark 4.1.1 (Structured Streaming)
- XGBoost and LSTM (ensemble model)
- Python 3.11, PyTorch, pandas

## How to run

**1. Start Kafka**
```powershell
docker compose -f docker/docker-compose.yml up -d
```
Wait about 30 seconds for Kafka to start up.

**2. Start the Spark streaming job** (keep this running)
```powershell
$env:SPARK_HOME = "E:\Big Data\spark"
$env:HADOOP_HOME = "E:\Big Data\hadoop"
.\.venv\Scripts\python src/03_spark_streaming_job.py --source kafka
```

**3. Run the producer** (in a second terminal)
```powershell
$env:SPARK_HOME = "E:\Big Data\spark"
$env:HADOOP_HOME = "E:\Big Data\hadoop"
.\.venv\Scripts\python src/03a_kafka_producer.py
```
This sends all the ICU data to Kafka. The streaming job picks it up and processes it in real time.

**4. Measure latency**
```powershell
.\.venv\Scripts\python src/03c_measure_latency.py
```

## Pipeline steps (in order)

| Script | What it does |
|--------|-------------|
| `01_prepare_cohort.py` | Labels patients using Sepsis-3 criteria |
| `02_build_stream_input.py` | Splits data into 50 streaming chunks |
| `03_spark_streaming_job.py` | Main streaming job (Kafka or CSV) |
| `03a_kafka_producer.py` | Sends data to Kafka topic |
| `03b_prepare_training_data.py` | Builds training features from stream output |
| `04_train_xgboost.py` | Trains XGBoost with 5-fold CV |
| `05_train_lstm.py` | Trains LSTM with 5-fold CV |
| `06_evaluate.py` | Evaluates ensemble and generates figures |

## Output
- `data/stream_output/` — ~28,890 windowed feature rows (136 patients)
- `data/processed/latency_results.json` — p50/p95/p99 latency stats
- `data/processed/paper_figures/` — figures used in the paper
- `data/processed/models/` — trained XGBoost and LSTM models

## Stop Kafka when done
```powershell
docker compose -f docker/docker-compose.yml down
```
