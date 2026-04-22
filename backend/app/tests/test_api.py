import pytest
from fastapi.testclient import TestClient
import os
import sys

# Set environment variables before importing the app
os.environ["MLFLOW_URI"]    = "http://localhost:5000"
os.environ["MODEL_NAME"]    = "sentiment-tfidf-lr"
os.environ["MODEL_STAGE"]   = "Production"
os.environ["PROCESSED_DIR"] = "airflow/data/processed"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from backend.app.main import app

client = TestClient(app)


def test_health():
    """Health endpoint returns 200 and status key."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_positive():
    """Positive review returns valid sentiment and confidence."""
    r = client.post("/api/v1/predict", json={"review": "This product is absolutely amazing!"})
    assert r.status_code == 200
    d = r.json()
    assert d["sentiment"] in ["Positive", "Negative", "Neutral"]
    assert 0 <= d["confidence"] <= 1
    assert "probabilities" in d


def test_predict_negative():
    """Negative review returns Negative or Neutral sentiment."""
    r = client.post("/api/v1/predict", json={"review": "Worst purchase ever, complete waste of money"})
    assert r.status_code == 200
    assert r.json()["sentiment"] in ["Negative", "Neutral"]


def test_predict_batch():
    """Batch of 15 reviews returns 15 predictions."""
    reviews = ["Great product!", "Terrible!", "It was okay"] * 5
    r = client.post("/api/v1/predict-batch", json={"reviews": reviews})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 15
    assert len(data["predictions"]) == 15


def test_predict_batch_too_large():
    """Batch of 1001 reviews returns 400 error."""
    reviews = ["test review"] * 1001
    r = client.post("/api/v1/predict-batch", json={"reviews": reviews})
    assert r.status_code == 400


def test_metrics_endpoint():
    """Prometheus metrics endpoint returns metric data."""
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_ready():
    """Readiness probe returns 200 or 503."""
    r = client.get("/api/v1/ready")
    assert r.status_code in [200, 503]