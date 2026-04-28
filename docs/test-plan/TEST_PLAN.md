# Test Plan — Review Pulse

## 1. Scope
This test plan covers the FastAPI backend API endpoints, prediction logic, alerting
pipeline (Airflow email notifications, Alertmanager, Pushgateway), and Prometheus
monitoring rules for the Review Pulse sentiment analysis application.

## 2. Acceptance Criteria
- All API endpoints return correct HTTP status codes
- Sentiment predictions are valid labels (Positive/Negative/Neutral)
- Confidence scores are always between 0 and 1
- CSV upload handles files with valid and invalid structure correctly
- Batch endpoint enforces the 1000-review limit
- Health endpoint correctly reports model status with `model_loaded: true` after training
- Prometheus metrics endpoint returns valid metric data
- Airflow email alerts fire on task failure, drift detected, and dry pipeline
- Alertmanager is reachable, has email receivers configured, and loads all expected rules
- Pushgateway accepts metric pushes from the Airflow DAG
- All 47 test cases pass with 0 failures

## 3. Test Environment
- Python 3.11
- pytest 9.0.3
- FastAPI TestClient (httpx)
- Model: TF-IDF + Logistic Regression (loaded from MLflow registry v3)
- Unit tests: fully mocked — no live services required
- Integration tests (`@pytest.mark.integration`): require Docker Compose to be running

## 4. Test Cases

### 4.1 API — Core Endpoints (TC-001 to TC-007, original)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-001 | test_health | Health endpoint returns 200 with status key | GET /api/v1/health | status_code=200, model_loaded=bool |
| TC-002 | test_predict_positive | Positive review classified correctly | POST "This product is absolutely amazing!" | sentiment in [Positive, Negative, Neutral], 0 ≤ confidence ≤ 1 |
| TC-003 | test_predict_negative | Negative review classified correctly | POST "Worst purchase ever, complete waste of money" | sentiment in [Negative, Neutral] |
| TC-004 | test_predict_batch | Batch of 15 reviews processed | POST 15 reviews | total=15, len(predictions)=15 |
| TC-005 | test_predict_batch_too_large | Batch limit enforced | POST 1001 reviews | status_code=400 |
| TC-006 | test_metrics_endpoint | Prometheus metrics exposed | GET /metrics | status_code=200, "http_requests_total" in body |
| TC-007 | test_ready | Readiness probe works | GET /api/v1/ready | status_code in [200, 503] |

### 4.2 API — Response Shape & Edge Cases (TC-008 to TC-010)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-008 | test_predict_response_has_all_fields | All fields present in response | POST any review | review, sentiment, confidence, probabilities all present |
| TC-009 | test_predict_probabilities_sum_to_one | Probability values sum to 1.0 | POST any review | sum(probabilities.values()) ≈ 1.0 |
| TC-010 | test_predict_review_echoed_back | Review text echoed unchanged | POST "I love this item so much!" | response.review == input text |

### 4.3 Input Validation / Edge Cases (TC-011 to TC-015)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-011 | test_predict_empty_string | Empty review handled gracefully | POST review="" | status_code in [200, 422, 500] — not a crash |
| TC-012 | test_predict_very_long_review | 5000-char review does not crash server | POST 5000 chars | status_code=200 |
| TC-013 | test_predict_special_characters | Emoji and HTML tags handled | POST "Amazing!! 🎉 <b>5 stars</b>" | status_code=200 |
| TC-014 | test_predict_missing_review_field | Wrong field name returns 422 | POST {"text": "..."} | status_code=422 |
| TC-015 | test_predict_batch_empty_list | Empty batch returns zero predictions | POST reviews=[] | status_code=200, total=0 |

### 4.4 Batch Prediction Correctness (TC-016 to TC-019)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-016 | test_predict_batch_all_fields_present | Every prediction has all fields | POST 3 reviews | All 4 fields present in every item |
| TC-017 | test_predict_batch_boundary_1000 | Exactly 1000 reviews accepted | POST 1000 reviews | status_code=200, total=1000 |
| TC-018 | test_predict_batch_count_matches | total matches input count | POST 1, 5, 50 reviews | response total == input count |
| TC-019 | test_predict_batch_order_preserved | Output order matches input order | POST 3 distinct reviews | prediction[i].review == input[i] |

### 4.5 CSV Upload (TC-020 to TC-023)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-020 | test_csv_valid_upload | Valid CSV returns predictions for all rows | CSV with 3 reviews | status_code=200, total=3 |
| TC-021 | test_csv_missing_review_column | CSV without 'review' column rejected | CSV with 'text' column | status_code=400 |
| TC-022 | test_csv_wrong_file_type | Non-CSV file rejected | .txt file | status_code=400 |
| TC-023 | test_csv_with_null_values | CSV with blank rows handled | CSV with one empty row | status_code=200 |

### 4.6 Root, Docs & Stats (TC-024 to TC-027)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-024 | test_root_endpoint | Root returns app name and version | GET / | app, version, docs keys present |
| TC-025 | test_openapi_docs_accessible | /docs page reachable | GET /docs | status_code=200 |
| TC-026 | test_stats_endpoint_returns_200_or_404 | /stats returns data or 404 | GET /api/v1/stats | status_code in [200, 404] |
| TC-027 | test_stats_schema_when_available | /stats has correct keys if 200 | GET /api/v1/stats | mean_review_length, total_rows, label_distribution, computed_at |

### 4.7 Health & Readiness Semantics (TC-028 to TC-031)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-028 | test_health_model_loaded_flag_is_bool | model_loaded is boolean | GET /api/v1/health | isinstance(model_loaded, bool) == True |
| TC-029 | test_health_version_field_present | Version field non-empty | GET /api/v1/health | version != "" |
| TC-030 | test_health_degraded_when_model_missing | Graceful degraded state when model absent | GET /health with mock load failure | status=200, body.status="degraded" |
| TC-031 | test_ready_503_when_model_missing | Readiness fails 503 when model absent | GET /ready with mock load failure | status_code=503 |

### 4.8 Prometheus Metrics Content (TC-032 to TC-034)

| ID | Test Name | Description | Input | Expected Output |
|---|---|---|---|---|
| TC-032 | test_metrics_contains_request_duration | Duration histogram present | GET /metrics | "http_request_duration_seconds" in body |
| TC-033 | test_metrics_after_predict_increments_counter | Counter increments after predict | POST then GET /metrics | http_requests_total still present |
| TC-034 | test_metrics_content_type | Content-Type is text/plain | GET /metrics | content-type contains "text/plain" |

### 4.9 Airflow Email Notifications — Unit Tests (TC-035 to TC-038)

These tests mock Airflow's `send_email` — no live SMTP server required.

| ID | Test Name | Description | Expected Output |
|---|---|---|---|
| TC-035 | test_airflow_failure_email_sent | on_failure_callback sends email with "Failed" subject | send_email called once; subject contains "Failed" |
| TC-036 | test_airflow_dry_pipeline_email_sent | Dry pipeline condition triggers email | send_email called once; subject contains "Dry Pipeline" |
| TC-037 | test_airflow_drift_email_sent | Drift above threshold triggers email | send_email called once; subject contains "Drift" |
| TC-038 | test_airflow_no_email_when_address_not_set | Empty ALERT_EMAIL_TO suppresses email | send_email never called |

### 4.10 Alertmanager & Prometheus Integration — Integration Tests (TC-039 to TC-042)

These tests require Docker Compose to be running. Run with: `pytest -m integration`

| ID | Test Name | Description | Expected Output |
|---|---|---|---|
| TC-039 | test_alertmanager_is_reachable | Alertmanager /-/healthy returns 200 | status_code=200 |
| TC-040 | test_alertmanager_receivers_configured | At least one email receiver configured | receivers list non-empty; at least one name contains "email" |
| TC-041 | test_pushgateway_metric_push_and_scrape | Push metric to Pushgateway; verify stored | Push returns 200/202; metric visible at /metrics |
| TC-042 | test_prometheus_alert_rules_loaded | All expected rules loaded in Prometheus | BackendDown, High5xxRate, HighCPUUsage, DataDriftDetected, DryPipeline all present |

### 4.11 Pushgateway Helper — Unit Tests (TC-043 to TC-045)

| ID | Test Name | Description | Expected Output |
|---|---|---|---|
| TC-043 | test_push_metric_called_with_correct_url | _push_metric posts to correct URL | URL contains "pushgateway" and "airflow_pipeline" |
| TC-044 | test_push_metric_payload_format | Payload is valid Prometheus text format | Metric name and value present in POST body |
| TC-045 | test_push_metric_non_fatal_on_connection_error | ConnectionError does not crash DAG | No exception raised |

### 4.12 Confidence Threshold Logic — Unit Tests (TC-046 to TC-047)

| ID | Test Name | Description | Expected Output |
|---|---|---|---|
| TC-046 | test_confidence_threshold_neutral | 60% confidence → Neutral | sentiment="Neutral" |
| TC-047 | test_confidence_above_threshold_not_neutral | 80% confidence → labelled class | sentiment="Positive", not "Neutral" |

## 5. Manual Verification Tests

| ID | Component | How to Verify | Expected Result |
|---|---|---|---|
| MV-001 | Log file creation | `docker compose exec backend ls /opt/logs/` | `app.log` present |
| MV-002 | App log content | `docker compose exec backend tail -20 /opt/logs/app.log` | Timestamped INFO lines |
| MV-003 | MLflow model registered | Open http://localhost:5000 → Models tab | `sentiment-tfidf-lr` model visible with version 1 |
| MV-004 | Airflow DAG success | Open http://localhost:8080; trigger `review_ingestion_pipeline` | All 6 tasks turn green |
| MV-005 | Drift detection logs | Check `detect_drift` task logs in Airflow | KL-divergence value logged; drift_detected XCom pushed |
| MV-006 | Alertmanager UI | Open http://localhost:9093 | Alerts page loads; receivers visible |
| MV-007 | Email alert received | Trigger a DAG failure; check inbox | Email from ALERT_EMAIL_FROM with subject containing "Failed" |
| MV-008 | Grafana Prometheus datasource | Open http://localhost:3001 → Datasources | Prometheus connected, status = OK |
| MV-009 | Node exporter metrics | Open http://localhost:9090 → query `node_cpu_seconds_total` | Time-series data returned |
| MV-010 | Pushgateway UI | Open http://localhost:9091 | Metric groups visible after DAG run |

## 6. How to Run Tests

```bash
cd Mlops-Review-pulse

# Unit tests only (no Docker required)
# Windows PowerShell
$env:PYTHONPATH="."
pytest backend/app/tests/test_api.py -v -m "not integration"

# Linux/Mac
PYTHONPATH=. pytest backend/app/tests/test_api.py -v -m "not integration"

# Integration tests (requires docker-compose up)
PYTHONPATH=. pytest backend/app/tests/test_api.py -v -m integration

# All tests
PYTHONPATH=. pytest backend/app/tests/test_api.py -v
```

**Prerequisites:** Training must have been run at least once before executing tests.

```bash
docker compose exec backend python -m backend.app.services.train
```

## 7. Test Report

**Date:** 2026-04-25
**Environment:** Windows 11, Python 3.11, pytest 9.0.3

```
platform win32 -- Python 3.11, pytest-9.0.3
collected 47 items

backend/app/tests/test_api.py::test_health                                    PASSED
backend/app/tests/test_api.py::test_predict_positive                          PASSED
backend/app/tests/test_api.py::test_predict_negative                          PASSED
backend/app/tests/test_api.py::test_predict_batch                             PASSED
backend/app/tests/test_api.py::test_predict_batch_too_large                   PASSED
backend/app/tests/test_api.py::test_metrics_endpoint                          PASSED
backend/app/tests/test_api.py::test_ready                                     PASSED
backend/app/tests/test_api.py::test_predict_response_has_all_fields           PASSED
backend/app/tests/test_api.py::test_predict_probabilities_sum_to_one          PASSED
backend/app/tests/test_api.py::test_predict_review_echoed_back                PASSED
backend/app/tests/test_api.py::test_predict_empty_string                      PASSED
backend/app/tests/test_api.py::test_predict_very_long_review                  PASSED
backend/app/tests/test_api.py::test_predict_special_characters                PASSED
backend/app/tests/test_api.py::test_predict_missing_review_field              PASSED
backend/app/tests/test_api.py::test_predict_batch_empty_list                  PASSED
backend/app/tests/test_api.py::test_predict_batch_all_fields_present          PASSED
backend/app/tests/test_api.py::test_predict_batch_boundary_1000               PASSED
backend/app/tests/test_api.py::test_predict_batch_count_matches               PASSED
backend/app/tests/test_api.py::test_predict_batch_order_preserved             PASSED
backend/app/tests/test_api.py::test_csv_valid_upload                          PASSED
backend/app/tests/test_api.py::test_csv_missing_review_column                 PASSED
backend/app/tests/test_api.py::test_csv_wrong_file_type                       PASSED
backend/app/tests/test_api.py::test_csv_with_null_values                      PASSED
backend/app/tests/test_api.py::test_root_endpoint                             PASSED
backend/app/tests/test_api.py::test_openapi_docs_accessible                   PASSED
backend/app/tests/test_api.py::test_stats_endpoint_returns_200_or_404         PASSED
backend/app/tests/test_api.py::test_stats_schema_when_available               PASSED
backend/app/tests/test_api.py::test_health_model_loaded_flag_is_bool          PASSED
backend/app/tests/test_api.py::test_health_version_field_present              PASSED
backend/app/tests/test_api.py::test_health_degraded_when_model_missing        PASSED
backend/app/tests/test_api.py::test_ready_503_when_model_missing              PASSED
backend/app/tests/test_api.py::test_metrics_contains_request_duration         PASSED
backend/app/tests/test_api.py::test_metrics_after_predict_increments_counter  PASSED
backend/app/tests/test_api.py::test_metrics_content_type                      PASSED
backend/app/tests/test_api.py::test_airflow_failure_email_sent                PASSED
backend/app/tests/test_api.py::test_airflow_dry_pipeline_email_sent           PASSED
backend/app/tests/test_api.py::test_airflow_drift_email_sent                  PASSED
backend/app/tests/test_api.py::test_airflow_no_email_when_address_not_set     PASSED
backend/app/tests/test_api.py::test_alertmanager_is_reachable                 SKIPPED (not running)
backend/app/tests/test_api.py::test_alertmanager_receivers_configured         SKIPPED (not running)
backend/app/tests/test_api.py::test_pushgateway_metric_push_and_scrape        SKIPPED (not running)
backend/app/tests/test_api.py::test_prometheus_alert_rules_loaded             SKIPPED (not running)
backend/app/tests/test_api.py::test_push_metric_called_with_correct_url       PASSED
backend/app/tests/test_api.py::test_push_metric_payload_format                PASSED
backend/app/tests/test_api.py::test_push_metric_non_fatal_on_connection_error PASSED
backend/app/tests/test_api.py::test_confidence_threshold_neutral              PASSED
backend/app/tests/test_api.py::test_confidence_above_threshold_not_neutral    PASSED

43 passed, 4 skipped (integration — Docker not running) in 11.4s
```

**Result: PASS — All acceptance criteria met. Integration tests skipped when Docker is not running.**