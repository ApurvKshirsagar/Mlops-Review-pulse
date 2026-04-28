import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import entropy

from airflow import DAG
from airflow.operators.python import PythonOperator
import airflow.utils.email as _airflow_email_mod

load_dotenv()

RAW_DIR          = os.environ.get("RAW_DIR",           "/opt/airflow/data/raw")
PROCESSED_DIR    = os.environ.get("PROCESSED_DIR",     "/opt/airflow/data/processed")
DRIFT_THRESHOLD  = float(os.environ.get("DRIFT_THRESHOLD", "0.1"))
RETRAIN_ON_DRIFT = os.environ.get("RETRAIN_ON_DRIFT",  "true").lower() == "true"
AIRFLOW_USER     = os.environ.get("AIRFLOW_ADMIN_USER",     "admin")
AIRFLOW_PASS     = os.environ.get("AIRFLOW_ADMIN_PASSWORD",  "admin")
ALERT_EMAIL      = os.environ.get("ALERT_EMAIL_TO",    "")
PROMETHEUS_PUSHGATEWAY = os.environ.get("PROMETHEUS_PUSHGATEWAY", "http://pushgateway:9091")

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _send_alert_email(subject: str, body: str) -> None:
    if not ALERT_EMAIL:
        logger.warning("ALERT_EMAIL_TO not set — skipping email notification")
        return
    try:
        _airflow_email_mod.send_email(to=ALERT_EMAIL, subject=subject, html_content=f"<pre>{body}</pre>")
        logger.info(f"Alert email sent to {ALERT_EMAIL}: {subject}")
    except Exception:
        logger.error("Failed to send alert email:\n" + traceback.format_exc())


def _push_metric(metric_name: str, value: float, labels: dict = None) -> None:
    try:
        label_str = ""
        if labels:
            label_str = "/" + "/".join(f"{k}/{v}" for k, v in labels.items())
        url = f"{PROMETHEUS_PUSHGATEWAY}/metrics/job/airflow_pipeline{label_str}"
        payload = f"# TYPE {metric_name} gauge\n{metric_name} {value}\n"
        resp = requests.post(url, data=payload, timeout=5)
        if resp.ok:
            logger.info(f"Pushed metric {metric_name}={value} to Pushgateway")
        else:
            logger.warning(f"Pushgateway push failed: {resp.status_code} {resp.text}")
    except Exception:
        logger.warning("Could not push metric to Pushgateway (non-fatal):\n" + traceback.format_exc())


# ── DAG callbacks ─────────────────────────────────────────────────────────────

def on_failure_callback(context):
    dag_id  = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    log_url = context.get("task_instance").log_url
    exc     = context.get("exception", "Unknown error")
    subject = f"[ReviewPulse] ❌ Task Failed: {dag_id}.{task_id}"
    body = (
        f"DAG:   {dag_id}\n"
        f"Task:  {task_id}\n"
        f"Time:  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Error: {exc}\n\n"
        f"Logs:  {log_url}\n"
    )
    _send_alert_email(subject, body)
    _push_metric("review_pulse_dag_last_run_failed", 1, {"dag": dag_id, "task": task_id})


def on_success_callback(context):
    dag_id = context.get("dag").dag_id
    _push_metric("review_pulse_dag_last_run_failed",       0,           {"dag": dag_id})
    _push_metric("review_pulse_dag_last_success_timestamp", time.time(), {"dag": dag_id})


# ── Tasks ─────────────────────────────────────────────────────────────────────

def ingest_data(**context):
    """Read UCI sentiment files, merge, and write raw_combined.csv.
    Also performs the Dry Pipeline check."""
    try:
        files = {
            "amazon": os.path.join(RAW_DIR, "sentiment labelled sentences/amazon_cells_labelled.txt"),
            "imdb":   os.path.join(RAW_DIR, "sentiment labelled sentences/imdb_labelled.txt"),
            "yelp":   os.path.join(RAW_DIR, "sentiment labelled sentences/yelp_labelled.txt"),
        }

        DRY_PIPELINE_HOURS = float(os.environ.get("DRY_PIPELINE_HOURS", "12"))
        stale_files = []
        for source, path in files.items():
            if not os.path.exists(path):
                stale_files.append(f"{source}: FILE MISSING ({path})")
            else:
                age_hours = (time.time() - os.path.getmtime(path)) / 3600
                if age_hours > DRY_PIPELINE_HOURS:
                    stale_files.append(f"{source}: last modified {age_hours:.1f}h ago")

        if stale_files:
            _send_alert_email(
                subject="[ReviewPulse] ⚠️ Dry Pipeline — No fresh data detected",
                body=(
                    f"No new data files detected within the {DRY_PIPELINE_HOURS}-hour window.\n\n"
                    + "\n".join(f"  • {s}" for s in stale_files)
                    + f"\n\nCheck: {RAW_DIR}\nAirflow: http://localhost:8080"
                ),
            )
            logger.warning(f"Dry pipeline detected: {stale_files}")

        frames = []
        for source, path in files.items():
            df = pd.read_csv(path, sep="\t", header=None, names=["review", "label"])
            df["source"] = source
            frames.append(df)
            logger.info(f"Loaded {len(df)} rows from '{source}'")

        combined = pd.concat(frames, ignore_index=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        combined.to_csv(os.path.join(PROCESSED_DIR, "raw_combined.csv"), index=False)
        logger.info(f"Ingestion complete — {len(combined)} rows")
        context["ti"].xcom_push(key="row_count", value=len(combined))

    except Exception:
        logger.error("ingest_data failed:\n" + traceback.format_exc())
        raise


def clean_data(**context):
    try:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "raw_combined.csv"))
        before = len(df)
        df["review"] = df["review"].str.lower()
        df["review"] = df["review"].str.replace(r"<.*?>", "", regex=True)
        df["review"] = df["review"].str.replace(r"[^a-z0-9\s]", "", regex=True)
        df["review"] = df["review"].str.strip()
        df.dropna(subset=["review"], inplace=True)
        df.drop_duplicates(subset=["review"], inplace=True)
        df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
        df.to_csv(os.path.join(PROCESSED_DIR, "cleaned.csv"), index=False)
        logger.info(f"Cleaned: {before} → {len(df)} rows")
    except Exception:
        logger.error("clean_data failed:\n" + traceback.format_exc())
        raise


def validate_data(**context):
    try:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned.csv"))
        assert {"review", "label", "sentiment", "source"}.issubset(df.columns)
        assert df["review"].isnull().sum() == 0
        assert df["sentiment"].isnull().sum() == 0
        assert len(df) >= 2000, f"Dataset too small: {len(df)} rows"
        logger.info(f"Validation passed ✓  rows={len(df)}")
    except AssertionError as exc:
        logger.error(f"Validation failed: {exc}")
        raise
    except Exception:
        logger.error("validate_data failed:\n" + traceback.format_exc())
        raise


def compute_baseline_stats(**context):
    try:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned.csv"))
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
        with open(os.path.join(PROCESSED_DIR, "baseline_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Baseline stats saved: {stats}")
    except Exception:
        logger.error("compute_baseline_stats failed:\n" + traceback.format_exc())
        raise


def detect_drift(**context):
    try:
        baseline_path = os.path.join(PROCESSED_DIR, "baseline_stats.json")
        if not os.path.exists(baseline_path):
            logger.warning("No baseline_stats.json — skipping drift check")
            context["ti"].xcom_push(key="drift_detected", value=False)
            return

        with open(baseline_path) as f:
            baseline = json.load(f)

        df = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned.csv"))
        df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))

        bins = np.linspace(0, 100, 21)
        baseline_lengths = np.random.normal(
            loc=baseline["mean_review_length"],
            scale=max(baseline["std_review_length"], 0.1),
            size=5000,
        )
        p, _ = np.histogram(baseline_lengths,    bins=bins, density=True)
        q, _ = np.histogram(df["review_length"], bins=bins, density=True)
        kl_div = float(entropy(p + 1e-10, q + 1e-10))
        drift_detected = kl_div > DRIFT_THRESHOLD

        logger.info(f"Drift — KL={kl_div:.4f}, threshold={DRIFT_THRESHOLD}, detected={drift_detected}")
        _push_metric("review_pulse_data_drift_kl_divergence", kl_div)

        if drift_detected:
            _send_alert_email(
                subject="[ReviewPulse] ⚠️ Data Drift Detected",
                body=(
                    f"Data drift detected.\n\n"
                    f"KL-divergence : {kl_div:.4f}\n"
                    f"Threshold     : {DRIFT_THRESHOLD}\n"
                    f"Time          : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                    f"Model will be retrained in the next task."
                ),
            )

        context["ti"].xcom_push(key="drift_detected", value=drift_detected)
        context["ti"].xcom_push(key="kl_divergence",  value=kl_div)

    except Exception:
        logger.error("detect_drift failed:\n" + traceback.format_exc())
        raise


# AFTER
def train_model(**context):
    """
    Task 6 — Train (or retrain) the sentiment model directly in the pipeline.
    All training logic is inlined to avoid importing the backend package,
    which is not installed inside the Airflow container.
    """
    try:
        drift_detected = context["ti"].xcom_pull(key="drift_detected", task_ids="detect_drift")
        is_first_run   = drift_detected is None

        if not is_first_run and not drift_detected and not RETRAIN_ON_DRIFT:
            logger.info("No drift detected and RETRAIN_ON_DRIFT=false — skipping training")
            return

        if drift_detected:
            logger.info("Drift detected — retraining model")
        else:
            logger.info("First pipeline run — training initial model")

        import pickle
        import mlflow
        import mlflow.sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        processed_dir = os.environ.get("PROCESSED_DIR", "/opt/airflow/data/processed")
        model_dir     = os.environ.get("MODEL_DIR",     "/opt/mlflow/artifacts/models")
        mlflow_uri    = os.environ.get("MLFLOW_URI",    "http://mlflow:5000")
        os.makedirs(model_dir, exist_ok=True)

        df = pd.read_csv(os.path.join(processed_dir, "cleaned.csv"))
        df = df.dropna(subset=["review", "sentiment"])
        logger.info(f"Loaded {len(df)} rows for training")

        X = df["review"].astype(str)
        y = df["sentiment"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
            ("lr",    LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy":   round(accuracy_score(y_test, y_pred), 4),
            "f1_macro":   round(f1_score(y_test, y_pred, average="macro"), 4),
            "precision":  round(precision_score(y_test, y_pred, average="macro"), 4),
            "recall":     round(recall_score(y_test, y_pred, average="macro"), 4),
            "test_size":  len(X_test),
            "train_size": len(X_train),
        }
        logger.info(f"Training metrics: {metrics}")

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("sentiment-analysis")
        with mlflow.start_run(run_name="tfidf-logistic-regression"):
            mlflow.log_params({
                "max_features": 10000, "ngram_range": "1,2", "min_df": 2,
                "C": 1.0, "max_iter": 1000, "solver": "lbfgs",
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="tfidf-lr-model",
                registered_model_name="sentiment-tfidf-lr",
            )
            run_id = mlflow.active_run().info.run_id

        model_path = os.path.join(model_dir, "tfidf_lr_pipeline.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Pickle saved to {model_path}")

        logger.info(f"Training complete — accuracy={metrics['accuracy']:.4f}, "
                    f"f1={metrics['f1_macro']:.4f}, run_id={run_id}")

        _push_metric("review_pulse_model_accuracy", metrics["accuracy"])
        _push_metric("review_pulse_model_f1",       metrics["f1_macro"])

        kl = context["ti"].xcom_pull(key="kl_divergence", task_ids="detect_drift") or 0.0
        _send_alert_email(
            subject="[ReviewPulse] ✅ Model Training Complete",
            body=(
                f"Model successfully trained and registered.\n\n"
                f"Accuracy      : {metrics['accuracy']:.4f}\n"
                f"F1 (macro)    : {metrics['f1_macro']:.4f}\n"
                f"MLflow run ID : {run_id}\n"
                f"KL-divergence : {kl:.4f}\n"
                f"Time          : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            ),
        )
        context["ti"].xcom_push(key="training_run_id", value=run_id)

    except Exception:
        logger.error("train_model failed:\n" + traceback.format_exc())
        raise

# ── DAG definition ────────────────────────────────────────────────────────────

default_args = {
    "owner":                     "review-pulse",
    "retries":                   2,
    "retry_delay":               timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay":           timedelta(minutes=30),
    "email_on_failure":          False,
    "email_on_retry":            False,
}

with DAG(
    dag_id="review_ingestion_pipeline",
    description="Ingest → clean → validate → baseline → drift-check → train",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["nlp", "ingestion", "training", "review-pulse"],
    on_failure_callback=on_failure_callback,
    on_success_callback=on_success_callback,
) as dag:

    t1 = PythonOperator(task_id="ingest_data",            python_callable=ingest_data,            provide_context=True)
    t2 = PythonOperator(task_id="clean_data",             python_callable=clean_data,             provide_context=True)
    t3 = PythonOperator(task_id="validate_data",          python_callable=validate_data,          provide_context=True)
    t4 = PythonOperator(task_id="compute_baseline_stats", python_callable=compute_baseline_stats, provide_context=True)
    t5 = PythonOperator(task_id="detect_drift",           python_callable=detect_drift,           provide_context=True)
    t6 = PythonOperator(task_id="train_model",            python_callable=train_model,            provide_context=True)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6