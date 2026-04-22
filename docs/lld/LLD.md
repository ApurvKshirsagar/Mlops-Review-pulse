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
Settings
  MLFLOW_URI: str       = env MLFLOW_URI        (default: http://mlflow:5000)
  MODEL_NAME: str       = env MODEL_NAME         (default: sentiment-tfidf-lr)
  MODEL_STAGE: str      = env MODEL_STAGE        (default: Production)
  PROCESSED_DIR: str    = env PROCESSED_DIR      (default: /opt/airflow/data/processed)
  APP_VERSION: str      = "1.0.0"
  APP_NAME: str         = "Review Pulse API"
```

### backend/app/core/model_loader.py
```
_model = None   (module-level cache)

load_model() -> sklearn.Pipeline
  1. If _model is cached, return it
  2. Try: mlflow.sklearn.load_model("models:/MODEL_NAME/MODEL_STAGE")
  3. If MLflow fails: fallback to local pickle at mlflow/artifacts/models/tfidf_lr_pipeline.pkl
  4. Cache result in _model
  5. Raise RuntimeError if both fail

get_model() -> sklearn.Pipeline
  Calls load_model(), returns cached model
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

Dependencies: task1 >> task2 >> task3 >> task4
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
| airflow-init | apache/airflow:2.9.1 | — | DB migration + admin user |
| airflow-scheduler | apache/airflow:2.9.1 | — | DAG scheduling |
| airflow-webserver | apache/airflow:2.9.1 | 8080 | Airflow UI |
| mlflow | ghcr.io/mlflow/mlflow:v3.11.1 | 5000 | Model registry + tracking |
| backend | mlops-review-pulse-backend | 8000 | FastAPI inference server |
| frontend | mlops-review-pulse-frontend | 3002 | Nginx static dashboard |
| prometheus | prom/prometheus:latest | 9090 | Metrics scraping |
| grafana | grafana/grafana:latest | 3001 | Metrics visualization |