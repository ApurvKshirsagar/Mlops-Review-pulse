import json
import logging
import os
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import entropy

from airflow import DAG
from airflow.operators.python import PythonOperator

# Load .env for local testing; no-op when vars are already in the environment
load_dotenv()

# ── Config — read from environment / .env ────────────────────────────────────
RAW_DIR         = os.environ.get("RAW_DIR",         "/opt/airflow/data/raw")
PROCESSED_DIR   = os.environ.get("PROCESSED_DIR",   "/opt/airflow/data/processed")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.1"))
RETRAIN_ON_DRIFT = os.environ.get("RETRAIN_ON_DRIFT", "true").lower() == "true"
AIRFLOW_USER    = os.environ.get("AIRFLOW_ADMIN_USER",     "admin")
AIRFLOW_PASS    = os.environ.get("AIRFLOW_ADMIN_PASSWORD",  "admin")

# Module-level logger — Airflow captures this in task logs
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Ingest raw data from multiple sources and merge into one CSV
# ─────────────────────────────────────────────────────────────────────────────
def ingest_data(**context):
    """
    Read the three UCI sentiment files, tag each row with its source,
    and save a combined raw CSV to PROCESSED_DIR/raw_combined.csv.
    Row count is pushed to XCom so downstream tasks can see it.
    """
    try:
        files = {
            "amazon": os.path.join(RAW_DIR, "sentiment labelled sentences/amazon_cells_labelled.txt"),
            "imdb":   os.path.join(RAW_DIR, "sentiment labelled sentences/imdb_labelled.txt"),
            "yelp":   os.path.join(RAW_DIR, "sentiment labelled sentences/yelp_labelled.txt"),
        }

        frames = []
        for source, path in files.items():
            # Files are tab-separated with no header: <review>\t<label>
            df = pd.read_csv(path, sep="\t", header=None, names=["review", "label"])
            df["source"] = source
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows from '{source}'")

        combined = pd.concat(frames, ignore_index=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        raw_path = os.path.join(PROCESSED_DIR, "raw_combined.csv")
        combined.to_csv(raw_path, index=False)

        logger.info(f"Ingestion complete — {len(combined)} total rows saved to {raw_path}")
        context["ti"].xcom_push(key="row_count", value=len(combined))

    except Exception:
        logger.error("ingest_data failed:\n" + traceback.format_exc())
        raise  # re-raise so Airflow marks this task as FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Clean and normalise text; add human-readable sentiment column
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(**context):
    """
    Apply standard NLP pre-processing:
      - lowercase
      - strip HTML tags
      - remove non-alphanumeric characters
      - drop nulls and duplicates
    Saves result to PROCESSED_DIR/cleaned.csv.
    """
    try:
        raw_path   = os.path.join(PROCESSED_DIR, "raw_combined.csv")
        clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")

        df = pd.read_csv(raw_path)
        before = len(df)

        df["review"] = df["review"].str.lower()
        df["review"] = df["review"].str.replace(r"<.*?>", "", regex=True)        # remove HTML
        df["review"] = df["review"].str.replace(r"[^a-z0-9\s]", "", regex=True) # remove punctuation
        df["review"] = df["review"].str.strip()

        df.dropna(subset=["review"], inplace=True)
        df.drop_duplicates(subset=["review"], inplace=True)

        logger.info(f"Dropped {before - len(df)} null/duplicate rows")

        # Map numeric labels to human-readable strings
        df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
        df.to_csv(clean_path, index=False)

        logger.info(f"Cleaned data saved — {len(df)} rows → {clean_path}")

    except Exception:
        logger.error("clean_data failed:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Schema and quality validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_data(**context):
    """
    Assert minimum data-quality requirements before training proceeds.
    Raises AssertionError (which Airflow treats as a task failure) on any violation.
    """
    try:
        clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
        df = pd.read_csv(clean_path)

        required_cols = {"review", "label", "sentiment", "source"}
        assert required_cols.issubset(df.columns), \
            f"Missing columns: {required_cols - set(df.columns)}"

        assert df["review"].isnull().sum() == 0,    "Null values found in 'review' column"
        assert df["sentiment"].isnull().sum() == 0, "Null values found in 'sentiment' column"
        assert len(df) >= 2000, f"Dataset too small: {len(df)} rows (minimum 2000)"

        dist = df["sentiment"].value_counts(normalize=True)
        logger.info(f"Label distribution (normalised):\n{dist}")
        logger.info("Validation passed ✓")

    except AssertionError as exc:
        logger.error(f"Validation failed: {exc}")
        raise
    except Exception:
        logger.error("validate_data failed:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Compute and save baseline statistics for drift detection
# ─────────────────────────────────────────────────────────────────────────────
def compute_baseline_stats(**context):
    """
    Compute review-length statistics and class distribution.
    Saves to PROCESSED_DIR/baseline_stats.json.
    These stats are used by the detect_drift task to measure future data drift.
    """
    try:
        clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
        df = pd.read_csv(clean_path)

        # Word count per review (used as the drift-detection signal)
        df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))

        stats = {
            "mean_review_length": round(df["review_length"].mean(), 2),
            "std_review_length":  round(df["review_length"].std(), 2),
            "min_review_length":  int(df["review_length"].min()),
            "max_review_length":  int(df["review_length"].max()),
            "total_rows":         len(df),
            "label_distribution": df["sentiment"].value_counts().to_dict(),
            "computed_at":        datetime.utcnow().isoformat(),
        }

        stats_path = os.path.join(PROCESSED_DIR, "baseline_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Baseline stats saved to {stats_path}: {stats}")

    except Exception:
        logger.error("compute_baseline_stats failed:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Detect data drift using KL-divergence
# ─────────────────────────────────────────────────────────────────────────────
def detect_drift(**context):
    """
    Compare the current batch's review-length distribution to the saved baseline
    using KL-divergence. Pushes 'drift_detected' (bool) to XCom.

    KL-divergence measures how much the current distribution has shifted from
    the reference (baseline). A value of 0 means identical distributions;
    higher values indicate more drift. DRIFT_THRESHOLD controls sensitivity.
    """
    try:
        baseline_path = os.path.join(PROCESSED_DIR, "baseline_stats.json")
        cleaned_path  = os.path.join(PROCESSED_DIR, "cleaned.csv")

        if not os.path.exists(baseline_path):
            logger.warning("No baseline_stats.json found — skipping drift check")
            context["ti"].xcom_push(key="drift_detected", value=False)
            return

        with open(baseline_path) as f:
            baseline = json.load(f)

        df = pd.read_csv(cleaned_path)
        df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))

        # Build matching histograms (0–100 words, 20 equal-width bins)
        bins = np.linspace(0, 100, 21)

        # Approximate the baseline distribution from its stored summary stats
        baseline_lengths = np.random.normal(
            loc=baseline["mean_review_length"],
            scale=max(baseline["std_review_length"], 0.1),  # avoid 0 std
            size=5000,
        )
        p, _ = np.histogram(baseline_lengths,       bins=bins, density=True)
        q, _ = np.histogram(df["review_length"],    bins=bins, density=True)

        # Add small epsilon to avoid log(0) in KL-divergence calculation
        p = p + 1e-10
        q = q + 1e-10

        kl_div = float(entropy(p, q))
        drift_detected = kl_div > DRIFT_THRESHOLD

        logger.info(
            f"Drift check — KL-divergence={kl_div:.4f}, "
            f"threshold={DRIFT_THRESHOLD}, drift_detected={drift_detected}"
        )
        if drift_detected:
            logger.warning(
                f"DATA DRIFT DETECTED — KL={kl_div:.4f} exceeds threshold {DRIFT_THRESHOLD}. "
                f"Model retraining will be triggered."
            )
        else:
            logger.info("No significant drift — model retraining not required")

        context["ti"].xcom_push(key="drift_detected", value=drift_detected)
        context["ti"].xcom_push(key="kl_divergence",  value=kl_div)

    except Exception:
        logger.error("detect_drift failed:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Conditionally trigger model retraining
# ─────────────────────────────────────────────────────────────────────────────
def trigger_retraining(**context):
    """
    Read the 'drift_detected' flag from XCom (set by detect_drift).
    If drift is detected AND RETRAIN_ON_DRIFT=true, call the Airflow REST API
    to trigger the model_training_pipeline DAG.
    If RETRAIN_ON_DRIFT=false, only logs a warning so a human can decide.
    """
    try:
        drift = context["ti"].xcom_pull(key="drift_detected", task_ids="detect_drift")

        if not drift:
            logger.info("No drift detected — skipping retraining")
            return

        if not RETRAIN_ON_DRIFT:
            logger.warning(
                "Drift detected but RETRAIN_ON_DRIFT=false — "
                "manual retraining required"
            )
            return

        logger.info("Triggering model_training_pipeline DAG via Airflow REST API...")

        # POST to the Airflow REST API to trigger the training DAG
        response = requests.post(
            "http://airflow-webserver:8080/api/v1/dags/model_training_pipeline/dagRuns",
            json={"conf": {"triggered_by": "drift_detection"}},
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=10,
        )

        if response.ok:
            logger.info(f"Retraining DAG triggered successfully: {response.json()}")
        else:
            logger.error(
                f"Failed to trigger retraining DAG: "
                f"status={response.status_code} body={response.text}"
            )

    except Exception:
        logger.error("trigger_retraining failed:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner":          "review-pulse",
    "retries":        1,
    "retry_delay":    timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="review_ingestion_pipeline",
    description="Ingest, clean, validate, baseline and drift-check UCI review data",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["nlp", "ingestion", "review-pulse"],
) as dag:

    t1 = PythonOperator(task_id="ingest_data",            python_callable=ingest_data,            provide_context=True)
    t2 = PythonOperator(task_id="clean_data",             python_callable=clean_data,             provide_context=True)
    t3 = PythonOperator(task_id="validate_data",          python_callable=validate_data,          provide_context=True)
    t4 = PythonOperator(task_id="compute_baseline_stats", python_callable=compute_baseline_stats, provide_context=True)
    t5 = PythonOperator(task_id="detect_drift",           python_callable=detect_drift,           provide_context=True)
    t6 = PythonOperator(task_id="trigger_retraining",     python_callable=trigger_retraining,     provide_context=True)

    # Linear execution order — each task depends on the previous succeeding
    t1 >> t2 >> t3 >> t4 >> t5 >> t6