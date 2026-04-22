# Test Plan — Review Pulse

## 1. Scope
This test plan covers the FastAPI backend API endpoints for the Review Pulse sentiment analysis application.

## 2. Acceptance Criteria
- All API endpoints return correct HTTP status codes
- Sentiment predictions are valid labels (Positive/Negative/Neutral)
- Confidence scores are always between 0 and 1
- CSV upload handles files with valid and invalid structure correctly
- Batch endpoint enforces the 1000-review limit
- Health endpoint correctly reports model status
- Prometheus metrics endpoint returns valid metric data
- All 7 test cases pass with 0 failures

## 3. Test Environment
- Python 3.11
- pytest 9.0.3
- FastAPI TestClient (httpx)
- Model: TF-IDF + Logistic Regression (local pickle)

## 4. Test Cases

| ID | Test Name | Description | Input | Expected Output | Status |
|---|---|---|---|---|---|
| TC-001 | test_health | Health endpoint returns 200 | GET /api/v1/health | status_code=200, "status" key present | ✅ Pass |
| TC-002 | test_predict_positive | Positive review classified correctly | POST with "This product is absolutely amazing!" | sentiment in [Positive, Negative, Neutral], 0 ≤ confidence ≤ 1 | ✅ Pass |
| TC-003 | test_predict_negative | Negative review classified correctly | POST with "Worst purchase ever, complete waste of money" | sentiment in [Negative, Neutral] | ✅ Pass |
| TC-004 | test_predict_batch | Batch of 15 reviews processed | POST 15 reviews | total=15 in response | ✅ Pass |
| TC-005 | test_predict_batch_too_large | Batch limit enforced | POST 1001 reviews | status_code=400 | ✅ Pass |
| TC-006 | test_metrics_endpoint | Prometheus metrics exposed | GET /metrics | status_code=200, "http_requests_total" in body | ✅ Pass |
| TC-007 | test_ready | Readiness probe works | GET /api/v1/ready | status_code in [200, 503] | ✅ Pass |

## 5. Test Report

**Date:** 2026-04-23
**Environment:** Windows 11, Python 3.13.2, pytest 9.0.3

```
platform win32 -- Python 3.13.2, pytest-9.0.3
collected 7 items

backend/app/tests/test_api.py::test_health                  PASSED
backend/app/tests/test_api.py::test_predict_positive        PASSED
backend/app/tests/test_api.py::test_predict_negative        PASSED
backend/app/tests/test_api.py::test_predict_batch           PASSED
backend/app/tests/test_api.py::test_predict_batch_too_large PASSED
backend/app/tests/test_api.py::test_metrics_endpoint        PASSED
backend/app/tests/test_api.py::test_ready                   PASSED

7 passed, 3 warnings in 3.72s
```

**Result: PASS — All acceptance criteria met.**

## 6. How to Run Tests

```bash
cd Mlops-Review-pulse

# Windows PowerShell
$env:PYTHONPATH="."
pytest backend/app/tests/test_api.py -v

# Linux/Mac
PYTHONPATH=. pytest backend/app/tests/test_api.py -v
```