# Review Pulse 
### Sentiment Analysis & Review Intelligence Platform

**Name:** Apurv Ravindra Kshirsagar  
**Roll No:** CE22B042

An end-to-end MLOps-grade web application that classifies customer reviews as Positive, Negative, or Neutral with confidence scores, visualizes sentiment trends, and exposes predictions via a REST API.

---

## Quick Start

```bash
git clone https://github.com/ApurvKshirsagar/Mlops-Review-pulse.git
cd Mlops-Review-pulse
dvc pull
docker compose up -d
docker compose ps
```

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Frontend Dashboard | http://localhost:3002 | — |
| FastAPI Swagger | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | admin/admin |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin/admin |

---

## Project Structure

```
Mlops-Review-pulse/
├── airflow/
│   ├── dags/review_pipeline.py     # Airflow DAG: ingest→clean→validate→baseline
│   └── data/                       # Raw + processed data (DVC tracked)
├── backend/
│   └── app/
│       ├── api/routes.py           # FastAPI endpoints
│       ├── core/                   # Config + model loader
│       ├── services/               # Predictor + trainer
│       └── main.py                 # App entry point
├── frontend/
│   └── index.html                  # Single-page dashboard
├── mlflow/                         # MLflow artifacts + DB
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/datasources.yml
├── dvc.yaml                        # DVC pipeline DAG
├── docker-compose.yml              # Full stack orchestration
└── .dvc/                           # DVC config
```

---

## ML Models

| Model | Accuracy | F1 (Macro) | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | 82.5% | 0.825 | Baseline, fast inference ~1.37ms |
| DistilBERT | 92.6% | 0.926 | Advanced, higher accuracy |

Both tracked in MLflow under experiment `sentiment-analysis`.

---

## MLOps Stack

| Component | Tool |
|---|---|
| Data Pipeline | Apache Airflow |
| Data Versioning | DVC + Git |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API Serving | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana |

---

## API Endpoints

```
GET  /api/v1/health          Health check
GET  /api/v1/ready           Readiness probe
POST /api/v1/predict         Single review prediction
POST /api/v1/predict-batch   Batch predictions (JSON)
POST /api/v1/predict-csv     CSV file upload
GET  /api/v1/stats           Dataset baseline stats
GET  /metrics                Prometheus metrics
```

---

## Dataset

UCI Sentiment Labelled Sentences — 3 sources, 3000 reviews total:
- `amazon_cells_labelled.txt` — Amazon product reviews
- `imdb_labelled.txt` — IMDB movie reviews
- `yelp_labelled.txt` — Yelp restaurant reviews

---

## Running Tests

```bash
$env:PYTHONPATH="."
pytest backend/app/tests/test_api.py -v
# 7 passed, 0 failed
```