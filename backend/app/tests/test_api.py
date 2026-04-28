"""
test_api.py — Review Pulse test suite
Run:  PYTHONPATH=. pytest backend/app/tests/test_api.py -v
      PYTHONPATH=. pytest backend/app/tests/test_api.py -v -m integration
"""

import importlib.util
import io
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

# ── Environment (must happen before importing the app) ───────────────────────
os.environ["MLFLOW_URI"]    = "http://localhost:5000"
os.environ["MODEL_NAME"]    = "sentiment-tfidf-lr"
os.environ["MODEL_STAGE"]   = "Production"
os.environ["PROCESSED_DIR"] = "airflow/data/processed"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from backend.app.main import app

# ── Shared mock model ─────────────────────────────────────────────────────────
# Prevents 500s when no trained model exists locally.
# Returns 85% confidence Positive for every review.
_mock_model = MagicMock()
_mock_model.predict_proba.return_value = [[0.15, 0.85]]
_mock_model.classes_ = ["Negative", "Positive"]


@pytest.fixture(autouse=True)
def _inject_model(request):
    """Patch get_model() for every test. Tests marked no_model_mock opt out."""
    if request.node.get_closest_marker("no_model_mock"):
        yield
        return
    with patch("backend.app.api.routes.get_model", return_value=_mock_model), \
         patch("backend.app.services.predictor.get_model", return_value=_mock_model):
        yield

# ── DAG loader — avoids importing Airflow on Windows ─────────────────────────
_DAG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../airflow/dags/review_pipeline.py")
)

def _load_dag():
    """Load the DAG file without triggering Airflow's DB/OS initialisation."""
    stubs = {
        "airflow":                   types.ModuleType("airflow"),
        "airflow.operators":         types.ModuleType("airflow.operators"),
        "airflow.operators.python":  types.ModuleType("airflow.operators.python"),
        "airflow.utils":             types.ModuleType("airflow.utils"),
        "airflow.utils.email":       types.ModuleType("airflow.utils.email"),
        "airflow.models":            types.ModuleType("airflow.models"),
    }
    stubs["airflow.utils.email"].send_email = MagicMock()
    stubs["airflow.operators.python"].PythonOperator = MagicMock()
    stubs["airflow"].DAG = MagicMock()
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

    spec = importlib.util.spec_from_file_location("review_pipeline", _DAG_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dag = _load_dag()
client = TestClient(app)


# ═════════════════════════════════════════════════════════════════════════════
# TC-001  Health
# ═════════════════════════════════════════════════════════════════════════════

def test_health():
    """TC-001: Health endpoint returns 200, status key, and boolean model_loaded."""
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert isinstance(body["model_loaded"], bool)
    assert body.get("version", "") != ""


# ═════════════════════════════════════════════════════════════════════════════
# TC-002 — TC-005  Single prediction
# ═════════════════════════════════════════════════════════════════════════════

def test_predict_valid_response_shape():
    """TC-002: Prediction returns all required fields with valid types."""
    r = client.post("/api/v1/predict", json={"review": "This product is absolutely amazing!"})
    assert r.status_code == 200
    d = r.json()
    assert d["sentiment"] in ["Positive", "Negative", "Neutral"]
    assert 0 <= d["confidence"] <= 1
    assert "probabilities" in d
    assert abs(sum(d["probabilities"].values()) - 1.0) < 0.01
    assert d["review"] == "This product is absolutely amazing!"


def test_predict_negative_review():
    """TC-003: Clearly negative review does not return Positive."""
    # Override mock to return negative confidence
    neg_model = MagicMock()
    neg_model.predict_proba.return_value = [[0.90, 0.10]]
    neg_model.classes_ = ["Negative", "Positive"]
    with patch("backend.app.api.routes.get_model", return_value=neg_model), patch("backend.app.services.predictor.get_model", return_value=neg_model):
        r = client.post("/api/v1/predict", json={"review": "Worst purchase ever, total waste of money"})
    assert r.status_code == 200
    assert r.json()["sentiment"] in ["Negative", "Neutral"]


def test_predict_missing_field_returns_422():
    """TC-004: Request with wrong field name returns 422 Unprocessable Entity."""
    r = client.post("/api/v1/predict", json={"text": "wrong field"})
    assert r.status_code == 422


def test_predict_very_long_review():
    """TC-005: 5000-char review does not crash the server."""
    r = client.post("/api/v1/predict", json={"review": "Great product! " * 333})
    assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════════════
# TC-006 — TC-009  Batch prediction
# ═════════════════════════════════════════════════════════════════════════════

def test_batch_returns_correct_count():
    """TC-006: Batch of N reviews returns total=N predictions in the same order."""
    reviews = ["Good product", "Bad product", "Average product"]
    r = client.post("/api/v1/predict-batch", json={"reviews": reviews})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    for i, pred in enumerate(data["predictions"]):
        assert pred["review"] == reviews[i]


def test_batch_limit_enforced():
    """TC-007: Batch of 1001 reviews returns 400; exactly 1000 is accepted."""
    assert client.post("/api/v1/predict-batch", json={"reviews": ["t"] * 1001}).status_code == 400
    assert client.post("/api/v1/predict-batch", json={"reviews": ["t"] * 1000}).status_code == 200


def test_batch_empty_list():
    """TC-008: Empty batch returns 200 with total=0."""
    r = client.post("/api/v1/predict-batch", json={"reviews": []})
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_batch_all_fields_present():
    """TC-009: Every item in batch response has the four required fields."""
    r = client.post("/api/v1/predict-batch", json={"reviews": ["Good", "Bad"]})
    assert r.status_code == 200
    for pred in r.json()["predictions"]:
        for field in ("review", "sentiment", "confidence", "probabilities"):
            assert field in pred


# ═════════════════════════════════════════════════════════════════════════════
# TC-010 — TC-012  CSV upload
# ═════════════════════════════════════════════════════════════════════════════

def test_csv_valid_upload():
    """TC-010: Valid CSV with 'review' column returns predictions for every row."""
    csv_data = "review\nGreat product\nTerrible quality\nIt was fine"
    r = client.post(
        "/api/v1/predict-csv",
        files={"file": ("reviews.csv", io.BytesIO(csv_data.encode()), "text/csv")},
    )
    assert r.status_code == 200
    assert r.json()["total"] == 3


def test_csv_missing_column_returns_400():
    """TC-011: CSV without 'review' column returns 400."""
    r = client.post(
        "/api/v1/predict-csv",
        files={"file": ("bad.csv", io.BytesIO(b"text,score\nGood,5"), "text/csv")},
    )
    assert r.status_code == 400


def test_csv_wrong_filetype_returns_400():
    """TC-012: Non-CSV file returns 400."""
    r = client.post(
        "/api/v1/predict-csv",
        files={"file": ("notes.txt", io.BytesIO(b"some text"), "text/plain")},
    )
    assert r.status_code == 400


# ═════════════════════════════════════════════════════════════════════════════
# TC-013 — TC-014  Readiness probe
# ═════════════════════════════════════════════════════════════════════════════

def test_ready_returns_200_when_model_loaded():
    """TC-013: /ready returns 200 when model is available."""
    r = client.get("/api/v1/ready")
    assert r.status_code == 200


@pytest.mark.no_model_mock
def test_ready_returns_503_when_model_missing():
    """TC-014: /ready returns 503 when model cannot be loaded."""
    with patch("backend.app.api.routes.get_model", side_effect=RuntimeError("No model")):
        r = client.get("/api/v1/ready")
    assert r.status_code == 503


# ═════════════════════════════════════════════════════════════════════════════
# TC-015  Prometheus metrics
# ═════════════════════════════════════════════════════════════════════════════

def test_metrics_endpoint():
    """TC-015: /metrics returns Prometheus text with expected metric names."""
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "http_request_duration_seconds" in r.text
    assert "text/plain" in r.headers.get("content-type", "")


# ═════════════════════════════════════════════════════════════════════════════
# TC-016 — TC-019  Airflow email alerts (unit — no live Airflow needed)
# ═════════════════════════════════════════════════════════════════════════════

def test_alert_email_sent_on_failure():
    """TC-016: _send_alert_email calls send_email with failure subject."""
    mock_send = MagicMock()
    sys.modules["airflow.utils.email"].send_email = mock_send
    _dag.ALERT_EMAIL = "apurv@gmail.com"

    _dag._send_alert_email("[ReviewPulse] ❌ Task Failed: ingest_data", "Error details here")

    mock_send.assert_called_once()
    subject = mock_send.call_args[1].get("subject") or mock_send.call_args[0][1]
    assert "Failed" in subject


def test_alert_email_sent_on_drift():
    """TC-017: _send_alert_email called with drift subject when KL > threshold."""
    mock_send = MagicMock()
    sys.modules["airflow.utils.email"].send_email = mock_send
    _dag.ALERT_EMAIL = "apurv@gmail.com"

    _dag._send_alert_email("[ReviewPulse] ⚠️ Data Drift Detected", "KL=0.23")

    mock_send.assert_called_once()
    subject = mock_send.call_args[1].get("subject") or mock_send.call_args[0][1]
    assert "Drift" in subject


def test_alert_email_sent_on_dry_pipeline():
    """TC-018: _send_alert_email called with Dry Pipeline subject."""
    mock_send = MagicMock()
    sys.modules["airflow.utils.email"].send_email = mock_send
    _dag.ALERT_EMAIL = "apurv@gmail.com"

    _dag._send_alert_email("[ReviewPulse] ⚠️ Dry Pipeline — No fresh data detected", "Files stale")

    mock_send.assert_called_once()
    subject = mock_send.call_args[1].get("subject") or mock_send.call_args[0][1]
    assert "Dry Pipeline" in subject


def test_no_email_when_address_not_configured():
    """TC-019: Empty ALERT_EMAIL suppresses send_email entirely."""
    mock_send = MagicMock()
    sys.modules["airflow.utils.email"].send_email = mock_send
    _dag.ALERT_EMAIL = ""

    _dag._send_alert_email("Subject", "Body")

    mock_send.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# TC-020 — TC-021  Pushgateway metric helper (unit)
# ═════════════════════════════════════════════════════════════════════════════

def test_push_metric_correct_url_and_payload():
    """TC-020: _push_metric posts to Pushgateway with correct URL and Prometheus payload."""
    captured = {}
    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return MagicMock(ok=True)

    with patch("requests.post", side_effect=fake_post):
        _dag.PROMETHEUS_PUSHGATEWAY = "http://pushgateway:9091"
        _dag._push_metric("review_pulse_data_drift_kl_divergence", 0.25)

    assert "pushgateway" in captured["url"]
    assert "airflow_pipeline" in captured["url"]
    assert "review_pulse_data_drift_kl_divergence" in captured["data"]
    assert "0.25" in captured["data"]


def test_push_metric_non_fatal_on_failure():
    """TC-021: _push_metric does NOT raise if Pushgateway is unreachable."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("refused")):
        _dag.PROMETHEUS_PUSHGATEWAY = "http://pushgateway:9091"
        _dag._push_metric("review_pulse_dag_last_run_failed", 1.0)  # must not throw


# ═════════════════════════════════════════════════════════════════════════════
# TC-022 — TC-023  Alertmanager integration (needs docker-compose up)
# ═════════════════════════════════════════════════════════════════════════════

ALERTMANAGER_URL = os.environ.get("ALERTMANAGER_URL", "http://localhost:9093")
PROMETHEUS_URL   = os.environ.get("PROMETHEUS_URL",   "http://localhost:9090")
PUSHGATEWAY_URL  = os.environ.get("PUSHGATEWAY_URL",  "http://localhost:9091")


@pytest.mark.integration
def test_alertmanager_healthy_and_has_email_receiver():
    """TC-022: Alertmanager is up and has at least one email receiver configured."""
    try:
        assert requests.get(f"{ALERTMANAGER_URL}/-/healthy", timeout=3).status_code == 200
        receivers = requests.get(f"{ALERTMANAGER_URL}/api/v2/receivers", timeout=3).json()
        names = [r.get("name", "") for r in receivers]
        assert any("email" in n for n in names), f"No email receiver. Got: {names}"
    except requests.exceptions.ConnectionError:
        pytest.skip("Alertmanager not running")


@pytest.mark.integration
def test_prometheus_alert_rules_loaded():
    """TC-023: Prometheus has loaded all expected alert rules from alert_rules.yml."""
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/rules", timeout=3)
        assert r.status_code == 200
        loaded = [
            rule["name"]
            for group in r.json().get("data", {}).get("groups", [])
            for rule in group.get("rules", [])
        ]
        for rule in ["BackendDown", "High5xxRate", "HighCPUUsage", "DataDriftDetected", "DryPipeline"]:
            assert rule in loaded, f"Missing alert rule: {rule}"
    except requests.exceptions.ConnectionError:
        pytest.skip("Prometheus not running")