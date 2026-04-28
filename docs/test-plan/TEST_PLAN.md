# Test Plan - MLOps Review Pulse

## Overview

This file describes testing done for the MLOps Review Pulse project based on the implemented test_api.py file.

The tests verify:

* Backend API endpoints
* Sentiment prediction
* Batch prediction
* CSV upload
* Health checks
* Metrics endpoint
* Alerting functions
* Monitoring integrations

---

## Test File Used

backend/app/tests/test_api.py

---

## Test Categories

### 1. Health Tests

| Test ID | Description                          |
| ------- | ------------------------------------ |
| TC-001  | /api/v1/health returns 200           |
| TC-002  | Contains status field                |
| TC-003  | Contains model_loaded boolean        |
| TC-004  | Contains version                     |
| TC-005  | /api/v1/ready returns correct status |

### 2. Single Prediction Tests

| Test ID | Description                           |
| ------- | ------------------------------------- |
| TC-006  | Valid review prediction works         |
| TC-007  | Negative review gives negative result |
| TC-008  | Missing field returns 422             |
| TC-009  | Very long review handled safely       |

### 3. Batch Prediction Tests

| Test ID | Description                   |
| ------- | ----------------------------- |
| TC-010  | Batch returns correct total   |
| TC-011  | Empty batch returns total = 0 |
| TC-012  | 1000 reviews accepted         |
| TC-013  | 1001 reviews rejected         |
| TC-014  | Output order preserved        |

### 4. CSV Upload Tests

| Test ID | Description                       |
| ------- | --------------------------------- |
| TC-015  | Valid CSV works                   |
| TC-016  | Missing review column returns 400 |
| TC-017  | Wrong file type returns 400       |

### 5. Metrics Tests

| Test ID | Description                   |
| ------- | ----------------------------- |
| TC-018  | /metrics returns 200          |
| TC-019  | Contains request count metric |
| TC-020  | Contains latency metric       |

### 6. Alert Tests

| Test ID | Description                      |
| ------- | -------------------------------- |
| TC-021  | Failure email alert sent         |
| TC-022  | Drift alert sent                 |
| TC-023  | Dry pipeline alert sent          |
| TC-024  | Empty email config sends nothing |

### 7. Pushgateway Tests

| Test ID | Description                        |
| ------- | ---------------------------------- |
| TC-025  | Metric pushed with correct payload |
| TC-026  | Connection failure does not crash  |

### 8. Integration Tests

| Test ID | Description                   |
| ------- | ----------------------------- |
| TC-027  | Alertmanager is healthy       |
| TC-028  | Email receiver exists         |
| TC-029  | Prometheus alert rules loaded |

---

## How to Run Tests

Activate Environment:
.\venv\Scripts\Activate.ps1

Run All Tests:
pytest backend/app/tests/test_api.py -v

Run Only Integration Tests:
pytest backend/app/tests/test_api.py -v -m integration

---

## Expected Result

22 passed, 1 skipped

---

## Final Status

PASS

The project backend, monitoring, alerting, and prediction APIs were tested successfully.
