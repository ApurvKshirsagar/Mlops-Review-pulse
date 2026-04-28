# Review Pulse
### Sentiment Analysis & Review Intelligence Platform

**Name:** Apurv Ravindra Kshirsagar  
**Roll No:** CE22B042

An end-to-end MLOps-grade web application that classifies customer reviews as Positive or Negative with confidence scores, visualizes sentiment trends, and exposes predictions via a REST API — with a full data versioning, pipeline orchestration, experiment tracking, and monitoring stack.

---

## Quick Start

```bash
git clone https://github.com/ApurvKshirsagar/Mlops-Review-pulse.git
cd Mlops-Review-pulse

# Copy env.example to .env and fill in your email/credentials
cp env.example .env

# Pull DVC-tracked data (processed CSVs + baseline stats)
dvc pull

# Start all services
docker compose up -d

# Verify all containers are healthy
docker compose ps
```

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Frontend Dashboard | http://localhost:3002 | — |
| FastAPI Swagger | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | admin / admin |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |

---

## Project Structure

```
Mlops-Review-pulse/
├── airflow/
│   ├── dags/review_pipeline.py     # Airflow DAG: ingest→clean→validate→baseline→drift
│   └── data/
│       ├── raw/                    # Source .txt files (DVC tracked)
│       └── processed/              # Cleaned CSVs + stats (DVC tracked)
├── backend/
│   └── app/
│       ├── api/routes.py           # FastAPI endpoints
│       ├── core/                   # Config, logging, model loader
│       ├── services/               # predictor.py + train.py
│       └── main.py                 # App entry point
├── frontend/
│   └── index.html                  # Single-page dashboard
├── mlflow/
│   └── artifacts/models/           # Saved model pickles
├── monitoring/
│   ├── prometheus/                 # prometheus.yml + alert_rules.yml
│   ├── grafana/                    # datasources.yml
│   └── alertmanager/               # alertmanager.yml
├── dvc.yaml                        # DVC pipeline definition
├── docker-compose.yml              # Full stack orchestration
└── .dvc/                           # DVC remote config
```

---

## MLOps Pipeline — How It Works

```
New Data (raw .txt files)
        │
        ▼
[1] Airflow DAG — review_ingestion_pipeline
        │  ingest_data            → merges 3 sources → raw_combined.csv
        │  clean_data             → normalise text   → cleaned.csv
        │  validate_data          → schema + size checks
        │  compute_baseline_stats → stats JSON
        └  detect_drift           → KL-divergence check → email alert if drift found
        │
        ▼
[2] Training (fully dockerised — one command)
        │  docker exec mlops-review-pulse-backend-1 python -m backend.app.services.train
        │  → trains TF-IDF + Logistic Regression
        │  → trains DistilBERT (if torch available)
        │  → logs metrics & artifacts to MLflow
        └  → promotes best model to Production in MLflow Registry
        │
        ▼
[3] Backend serves predictions via FastAPI
        │
        ▼
[4] Prometheus scrapes metrics → Grafana dashboard
```

---

## Adding New Data & Running the Full Pipeline

### Step 1 — Add new raw data files

Place your new `.txt` files (tab-separated: `review\tlabel`) in:

```
airflow/data/raw/sentiment labelled sentences/
```

Supported filenames (replace or add alongside existing):
```
amazon_cells_labelled.txt
imdb_labelled.txt
yelp_labelled.txt
```

### Step 2 — Run the Airflow data pipeline

**Option A — via Airflow UI (recommended)**
1. Open http://localhost:8080
2. Find `review_ingestion_pipeline`
3. Click the ▶ **Trigger DAG** button
4. Watch all 5 nodes turn green: `ingest → clean → validate → baseline → drift`

**Option B — via DVC from terminal**

```bash
dvc repro
```

This runs each pipeline stage in order and only re-runs stages whose inputs have changed. Outputs (`raw_combined.csv`, `cleaned.csv`, `baseline_stats.json`) are cached by DVC.

> **Note:** `dvc repro` calls into the running Docker container. Make sure `docker compose up -d` is running first.

### Step 3 — Train the model (fully dockerised)

Training runs inside the backend container — no local Python environment or environment variables needed. The container already has `PYTHONPATH`, `PROCESSED_DIR`, `MLFLOW_URI`, and `MODEL_DIR` configured via `docker-compose.yml`, and all `.env` values are injected automatically.

```bash
docker exec mlops-review-pulse-backend-1 python -m backend.app.services.train
```

This will:
- Load `cleaned.csv` from the processed data volume
- Train TF-IDF + Logistic Regression and DistilBERT
- Log parameters, metrics, and artifacts to MLflow at http://localhost:5000
- Register and promote the best model to `Production` in the MLflow Model Registry
- Push a Git tag for the winning model version (e.g. `model/sentiment-distilbert/v3`)
- Save a comparison summary to `mlflow/artifacts/models/model_comparison.json`

### Step 4 — Commit data versions to DVC

```bash
dvc add airflow/data/processed/cleaned.csv
dvc push
git add airflow/data/processed/cleaned.csv.dvc
git commit -m "data: update processed dataset"
git push
```

---

## ML Models

| Model | Accuracy | F1 (Macro) | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | 82.5% | 0.825 | Lightweight baseline, always trained |
| DistilBERT | 93.2% | 0.932 | Higher accuracy, requires torch |

Both experiments are tracked in MLflow under the `sentiment-analysis` experiment. The best model is automatically promoted to the `Production` stage in the MLflow Model Registry and tagged in Git.

---

## MLOps Stack

| Component | Tool | Purpose |
|---|---|---|
| Data Pipeline | Apache Airflow 2.9.1 | Orchestrate ingest → drift |
| Data Versioning | DVC + Git | Track datasets & processed files |
| Experiment Tracking | MLflow v3 | Log metrics, params, artifacts |
| Model Registry | MLflow Model Registry | Version & stage models |
| API Serving | FastAPI + Uvicorn | Serve predictions |
| Containerization | Docker + Docker Compose | Full stack reproducibility |
| Metrics Collection | Prometheus + Pushgateway | Scrape pipeline & API metrics |
| Dashboards | Grafana | Visualize pipeline health |
| Alerting | Prometheus Alertmanager + Email | Drift & failure notifications |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/ready` | Readiness probe (503 if no model) |
| POST | `/api/v1/predict` | Single review → sentiment + confidence |
| POST | `/api/v1/predict-batch` | Batch predictions (JSON array) |
| POST | `/api/v1/predict-csv` | CSV file upload, returns predictions |
| GET | `/api/v1/stats` | Baseline dataset statistics |
| GET | `/metrics` | Prometheus metrics scrape endpoint |

**Example predict call:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is absolutely amazing!"}'
```

---

## Dataset

UCI Sentiment Labelled Sentences — 3 sources, 3000 reviews total:

| File | Source | Count |
|---|---|---|
| `amazon_cells_labelled.txt` | Amazon product reviews | 1000 |
| `imdb_labelled.txt` | IMDB movie reviews | 1000 |
| `yelp_labelled.txt` | Yelp restaurant reviews | 1000 |

Format: tab-separated, `review\tlabel` where label is `0` (Negative) or `1` (Positive).

---

## Running Tests

```bash
docker exec mlops-review-pulse-backend-1 pytest backend/app/tests/test_api.py -v
```

Expected output: **22 passed, 1 skipped** (Alertmanager integration test skipped when not running).

---

## Email Alerts

Configure in `.env` to receive alerts for:
- **Dry Pipeline** — data files not updated within `DRY_PIPELINE_HOURS`
- **Data Drift Detected** — KL-divergence exceeds `DRIFT_THRESHOLD`
- **Task Failure** — any Airflow task fails
- **Training Complete** — model trained and registered successfully

```env
ALERT_EMAIL_TO=your@email.com
ALERT_EMAIL_FROM=your.gmail@gmail.com
ALERT_EMAIL_PASSWORD=xxxx-xxxx-xxxx-xxxx   # Gmail App Password
AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
AIRFLOW__SMTP__SMTP_USER=your.gmail@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

> Generate a Gmail App Password at: https://myaccount.google.com/apppasswords

---

## Stopping the Stack

```bash
docker compose down        # Stop containers, keep volumes
docker compose down -v     # Stop and delete all volumes (full reset)
```