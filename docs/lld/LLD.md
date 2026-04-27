# Low-Level Design — Review Pulse API

## 1. API Endpoint Specifications

### GET /api/v1/health
**Description:** Health check and model status
**Response 200:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### GET /api/v1/ready
**Description:** Readiness probe for orchestration health checks
**Response 200:** `{"status": "ready"}`
**Response 503:** `{"detail": "No model available. Run training first."}`

### POST /api/v1/predict
**Description:** Predict sentiment for a single review
**Content-Type:** `application/json`
**Request Body:**
```json
{
  "review": "This product is absolutely amazing!"
}
```
**Response 200:**
```json
{
  "review": "This product is absolutely amazing!",
  "sentiment": "Positive",
  "confidence": 0.9134,
  "probabilities": {
    "Negative": 0.0866,
    "Positive": 0.9134
  }
}
```
**Response 500:** `{"detail": "<error message>"}`

### POST /api/v1/predict-batch
**Description:** Predict sentiment for a list of reviews
**Content-Type:** `application/json`
**Constraints:** Maximum 1000 reviews per request
**Request Body:**
```json
{
  "reviews": ["Great product!", "Terrible experience", "It was okay"]
}
```
**Response 200:**
```json
{
  "predictions": [
    {
      "review": "Great product!",
      "sentiment": "Positive",
      "confidence": 0.87,
      "probabilities": {"Negative": 0.13, "Positive": 0.87}
    }
  ],
  "total": 3
}
```
**Response 400:** `{"detail": "Max 1000 reviews per request"}`

### POST /api/v1/predict-csv
**Description:** Upload a CSV file for bulk sentiment prediction
**Content-Type:** `multipart/form-data`
**Field name:** `file`
**CSV Requirements:** Must contain a column named exactly `review`
**Response:** Same schema as predict-batch
**Response 400:** `{"detail": "Only CSV files accepted"}` or `{"detail": "CSV must have a 'review' column"}`

### GET /api/v1/stats
**Description:** Return dataset baseline statistics from the data pipeline
**Response 200:**
```json
{
  "mean_review_length": 11.42,
  "std_review_length": 7.31,
  "min_review_length": 1,
  "max_review_length": 52,
  "total_rows": 2717,
  "label_distribution": {"Positive": 1371, "Negative": 1346},
  "computed_at": "2026-04-22T16:33:10.123456"
}
```

### GET /metrics
**Description:** Prometheus metrics endpoint
**Content-Type:** `text/plain; version=1.0.0`
**Sample output:**
```
http_requests_total{handler="/api/v1/predict",method="POST",status_code="200"} 1776.0
http_request_duration_seconds_bucket{le="0.005",handler="/api/v1/predict"} 1750.0
```

---

## 2. Module Design

### backend/app/core/config.py
```
Settings (all values loaded from environment variables via python-dotenv)
  MLFLOW_URI: str       = env MLFLOW_URI        (default: http://mlflow:5000)
  MODEL_NAME: str       = env MODEL_NAME         (default: sentiment-tfidf-lr)
  MODEL_STAGE: str      = env MODEL_STAGE        (default: Production)
  PROCESSED_DIR: str    = env PROCESSED_DIR      (default: /opt/airflow/data/processed)
  LOG_DIR: str          = env LOG_DIR            (default: /opt/logs)
  APP_VERSION: str      = "1.0.0"
  APP_NAME: str         = "Review Pulse API"
  DRIFT_THRESHOLD: float = env DRIFT_THRESHOLD   (default: 0.1)
  RETRAIN_ON_DRIFT: bool = env RETRAIN_ON_DRIFT  (default: true)
```

### backend/app/core/logging_config.py
```
get_logger(name: str) -> logging.Logger
  Configures a named logger with two handlers (added once only):
    - StreamHandler (INFO+) → stdout, captured by Docker
    - RotatingFileHandler (DEBUG+) → LOG_DIR/app.log, 10MB max, 5 backups
  LOG_DIR is a host-mounted volume — logs are never stored inside the image.
```

### backend/app/core/model_loader.py
```
_model = None   (module-level singleton cache)

load_model() -> sklearn.Pipeline
  1. If _model is cached, return immediately
  2. Try MLflow registry (MLflow v3 compatible):
       client = MlflowClient(tracking_uri=MLFLOW_URI)
       versions = client.get_latest_versions(MODEL_NAME)
       model_uri = f"models:/{MODEL_NAME}/{versions[-1].version}"
       _model = mlflow.sklearn.load_model(model_uri)
  3. If MLflow fails: fallback to local pickle at MODEL_DIR/tfidf_lr_pipeline.pkl
  4. Cache result in _model
  5. Raise RuntimeError if both strategies fail

  Note: MLflow v3 removed named stages (Production/Staging).
  Models are now loaded by version number. get_latest_versions() returns
  all versions sorted ascending; [-1] gives the most recently registered.

get_model() -> sklearn.Pipeline
  Calls load_model(), returns cached model

reload_model() -> sklearn.Pipeline
  Clears _model cache and calls load_model() — used after training
```

### backend/app/services/predictor.py
```
CONFIDENCE_THRESHOLD = 0.65

predict_single(review: str) -> dict
  1. model = get_model()
  2. proba = model.predict_proba([review])[0]
  3. max_conf = max(proba)
  4. pred_class = classes[argmax(proba)]
  5. if max_conf < CONFIDENCE_THRESHOLD: sentiment = "Neutral"
     else: sentiment = pred_class
  6. return {review, sentiment, confidence, probabilities}

predict_batch(reviews: list[str]) -> list[dict]
  return [predict_single(r) for r in reviews]
```

### backend/app/services/train.py
```
Key functions:

load_data() -> pd.DataFrame
  Reads PROCESSED_DIR/cleaned.csv, drops null/invalid rows

train_tfidf_lr(df) -> (run_id, metrics)
  Trains TF-IDF (10k features, bigrams) + LogisticRegression pipeline
  Logs params, metrics, classification report to MLflow
  Saves local pickle to MODEL_DIR/tfidf_lr_pipeline.pkl
  Calls _log_training_event() to append JSON line to LOG_DIR/training.log

train_distilbert(df) -> (run_id, metrics) | (None, None)
  Fine-tunes distilbert-base-uncased for binary classification
  Skips gracefully if torch/transformers not installed

compare_and_register_best(lr_metrics, bert_metrics, ...) -> str
  Compares F1-macro scores, promotes winner to MLflow registry
  Sets model alias "production" via client.set_registered_model_alias()
  Calls _create_git_tag() to create annotated git tag

_log_training_event(model, run_id, metrics)
  Appends one JSON line to LOG_DIR/training.log:
  {"timestamp", "model", "run_id", "accuracy", "f1_macro", ...}

_create_git_tag(model, version, metrics)
  Runs: git tag -a model/<name>/v<version> -m "..."
  Non-fatal — logs warning if git unavailable
```

### backend/app/api/routes.py
```
Router prefix: /api/v1

GET  /health          → HealthResponse
GET  /ready           → {"status": "ready"} | 503
POST /predict         → SinglePredictResponse
POST /predict-batch   → BatchPredictResponse
POST /predict-csv     → BatchPredictResponse
GET  /stats           → dict (from baseline_stats.json)
```

---

## 3. Airflow DAG — review_ingestion_pipeline

```
DAG ID:           review_ingestion_pipeline
Schedule:         @once
Start Date:       2024-01-01
Catchup:          False
Tags:             [nlp, ingestion, review-pulse]

Task 1: ingest_data
  Input:  3 raw .txt files from UCI dataset
  Output: /opt/airflow/data/processed/raw_combined.csv
  XCom:   pushes row_count

Task 2: clean_data
  Input:  raw_combined.csv
  Steps:  lowercase, strip HTML, remove special chars, drop nulls/dupes
  Output: /opt/airflow/data/processed/cleaned.csv

Task 3: validate_data
  Input:  cleaned.csv
  Checks: required columns, no nulls, >= 2000 rows, class balance >= 10%
  Fails:  AssertionError if any check fails

Task 4: compute_baseline_stats
  Input:  cleaned.csv
  Output: /opt/airflow/data/processed/baseline_stats.json
  Stats:  mean/std/min/max review length, label distribution, timestamp

Task 5: detect_drift
  Input:  cleaned.csv + baseline_stats.json
  Method: KL-divergence on review word-count histograms (20 bins, 0-100 words)
  XCom:   pushes drift_detected (bool) and kl_divergence (float)
  Logs:   KL value and whether threshold was exceeded

Task 6: trigger_retraining
  Input:  drift_detected XCom from task 5
  Action: If drift_detected=True AND RETRAIN_ON_DRIFT=true,
          POSTs to Airflow REST API to trigger model_training_pipeline DAG
  Skip:   If no drift, or RETRAIN_ON_DRIFT=false (logs warning only)

Dependencies: t1 >> t2 >> t3 >> t4 >> t5 >> t6
```

---

## 4. MLflow Experiment Schema

**Experiment name:** `sentiment-analysis`

### Run: tfidf-logistic-regression
| Category | Key | Value |
|---|---|---|
| Parameters | max_features | 10000 |
| Parameters | ngram_range | 1,2 |
| Parameters | C | 1.0 |
| Parameters | solver | lbfgs |
| Metrics | accuracy | 0.8254 |
| Metrics | f1_macro | 0.8254 |
| Metrics | precision | 0.8256 |
| Metrics | recall | 0.8255 |
| Metrics | train_size | 2173 |
| Metrics | test_size | 544 |
| Artifacts | tfidf-lr-model/ | sklearn Pipeline |
| Artifacts | tfidf_lr_report.json | classification report |

### Run: distilbert-classifier
| Category | Key | Value |
|---|---|---|
| Parameters | model | distilbert-base-uncased |
| Parameters | num_train_epochs | 2 |
| Parameters | per_device_train_batch | 16 |
| Parameters | max_length | 128 |
| Metrics | accuracy | 0.9191 |
| Metrics | f1_macro | 0.9190 |
| Artifacts | distilbert-model/ | transformers Pipeline |

---

## 5. DVC Pipeline (dvc.yaml)

```yaml
stages:
  ingest:
    deps: [amazon_cells_labelled.txt, imdb_labelled.txt, yelp_labelled.txt]
    outs: [raw_combined.csv]

  clean:
    deps: [raw_combined.csv]
    outs: [cleaned.csv]

  validate:
    deps: [cleaned.csv]

  baseline_stats:
    deps: [cleaned.csv]
    outs: [baseline_stats.json]
```

DAG: ingest → clean → validate
                    ↘ baseline_stats

---

## 6. Docker Compose Services

| Service | Image | Port | Purpose |
|---|---|---|---|
| postgres | postgres:13 | 5432 | Airflow metadata DB |
| airflow-init | apache/airflow:2.9.1 | — | One-time: DB migration + admin user creation |
| airflow-scheduler | apache/airflow:2.9.1 | — | DAG scheduling and task execution |
| airflow-webserver | apache/airflow:2.9.1 | 8080 | Airflow UI |
| mlflow | ghcr.io/mlflow/mlflow:v3.11.1 | 5000 | Model registry + experiment tracking |
| backend | mlops-review-pulse-backend | 8000 | FastAPI inference server |
| frontend | mlops-review-pulse-frontend | 3002 | Nginx static dashboard |
| prometheus | prom/prometheus:latest | 9090 | Metrics scraping |
| grafana | grafana/grafana:latest | 3001 | Metrics visualization |

**Note on MLflow v3:** The server is started with `--allowed-hosts mlflow,localhost,127.0.0.1,backend` to permit internal Docker service-name requests. Without this flag, MLflow v3's security middleware rejects calls from the backend container with "Invalid Host header".