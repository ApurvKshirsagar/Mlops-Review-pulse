    # High-Level Design — Review Pulse

## 1. Problem Statement

Businesses receive thousands of customer reviews but cannot process them at scale. Recurring complaints and sentiment shifts go unnoticed, leading to poor product decisions.

## 2. Solution Overview

An end-to-end AI web application that:
- Accepts single reviews or bulk CSV uploads
- Classifies each as Positive / Negative / Neutral with confidence score
- Displays interactive sentiment trend dashboard
- Exposes REST API for programmatic access

## 3. Architecture Decision Record

### 3.1 Frontend–Backend Decoupling
**Decision:** Strict separation via REST API only.
**Rationale:** Allows independent scaling, independent deployment, and clean interface contract. Frontend is pure static HTML/JS served by Nginx with no server-side logic.

### 3.2 Model Selection
**Decision:** TF-IDF + LR as baseline, DistilBERT as advanced variant.
**Rationale:** TF-IDF LR is fast (~1.37ms inference), interpretable, and achieves 82.5% F1. DistilBERT achieves 92.6% F1 at the cost of higher inference latency. Both are compared in MLflow to justify the production choice.

### 3.3 MLflow for Model Registry
**Decision:** All models registered in MLflow, loaded via `models:/name/Production` URI.
**Rationale:** Enables zero-downtime model promotion, rollback, and A/B testing without code changes. Decouples model lifecycle from application deployment.

### 3.4 Airflow for Data Pipeline
**Decision:** Apache Airflow DAG for ingestion → cleaning → validation → baseline stats.
**Rationale:** Provides visual pipeline monitoring, task retry logic, XCom for inter-task communication, and scheduling capability for future automation.

### 3.5 DVC for Data Versioning
**Decision:** DVC tracks raw data files and processed outputs separately from code.
**Rationale:** Enables full reproducibility — any Git commit can be paired with a DVC commit to reproduce the exact data + model state used in that experiment.

### 3.6 Docker Compose for Deployment
**Decision:** All 9 services containerized, orchestrated with Docker Compose.
**Rationale:** Ensures environment parity between development and production. Single `docker compose up -d` boots the entire stack with no manual steps.

### 3.7 Neutral Class via Confidence Thresholding
**Decision:** Reviews with max class confidence < 0.65 are labelled "Neutral".
**Rationale:** The UCI dataset is binary (Positive/Negative only). Real-world reviews are often ambiguous. Confidence thresholding is a pragmatic way to introduce a Neutral class without retraining.

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP :3002
┌──────────────────────────▼──────────────────────────────────┐
│              Frontend (Nginx + HTML/CSS/JS)                  │
│         Single Review | CSV Upload | Dashboard               │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API :8000
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                            │
│    /predict  /predict-batch  /predict-csv  /health           │
│              Prometheus Instrumentation /metrics             │
└────────┬──────────────────┬────────────────────────────────┘
         │                  │
         │ Load Model       │ Scrape Metrics
┌────────▼──────┐  ┌────────▼──────────────────────────────┐
│  MLflow       │  │  Prometheus (:9090) → Grafana (:3001)  │
│  Model Reg.   │  │  Request rate, latency, error rate     │
│  (:5000)      │  └────────────────────────────────────────┘
└───────────────┘

Data Pipeline:
┌────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐
│  UCI   │───▶│ Airflow DAG  │───▶│   DVC    │───▶│   MLflow     │
│  Data  │    │ 4-task pipe  │    │Versioned │    │  Experiment  │
└────────┘    └──────────────┘    └──────────┘    └──────────────┘

Docker Compose Services:
postgres | airflow-init | airflow-scheduler | airflow-webserver
mlflow | backend | frontend | prometheus | grafana
```

## 5. Data Flow

1. User uploads CSV → Frontend → POST /predict-csv → Backend
2. Backend loads model from MLflow registry (cached in memory after first load)
3. Predictor applies TF-IDF vectorization → Logistic Regression → returns label + confidence
4. If confidence < 0.65 → labelled "Neutral"
5. Response JSON → Frontend renders charts + table + keyword analysis

## 6. Non-Functional Requirements

| Requirement | Target | Achieved |
|---|---|---|
| Inference latency | < 200ms | ~1.37ms avg (TF-IDF LR) |
| Error rate | < 5% | 0% in all tests |
| Model F1 score | > 80% | 82.5% (LR), 92.6% (DistilBERT) |
| Deployment | Single command | `docker compose up -d` |
| Test coverage | > 7 test cases | 7 passing, 0 failing |
| Monitoring | Real-time metrics | Prometheus + Grafana live dashboard |