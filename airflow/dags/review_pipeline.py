from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import re
import os
import logging
import json

logger = logging.getLogger(__name__)

RAW_DIR       = "/opt/airflow/data/raw"
PROCESSED_DIR = "/opt/airflow/data/processed"

def ingest_data(**context):
    files = {
        "amazon": os.path.join(RAW_DIR, "sentiment labelled sentences/amazon_cells_labelled.txt"),
        "imdb":   os.path.join(RAW_DIR, "sentiment labelled sentences/imdb_labelled.txt"),
        "yelp":   os.path.join(RAW_DIR, "sentiment labelled sentences/yelp_labelled.txt"),
    }
    frames = []
    for source, path in files.items():
        df = pd.read_csv(path, sep="\t", header=None, names=["review", "label"])
        df["source"] = source
        frames.append(df)
        logger.info(f"Loaded {len(df)} rows from {source}")

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_path = os.path.join(PROCESSED_DIR, "raw_combined.csv")
    combined.to_csv(raw_path, index=False)
    logger.info(f"Ingestion complete — {len(combined)} total rows")
    context["ti"].xcom_push(key="row_count", value=len(combined))


def clean_data(**context):
    raw_path   = os.path.join(PROCESSED_DIR, "raw_combined.csv")
    clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")

    df = pd.read_csv(raw_path)
    df["review"] = df["review"].str.lower()
    df["review"] = df["review"].str.replace(r"<.*?>", "", regex=True)
    df["review"] = df["review"].str.replace(r"[^a-z0-9\s]", "", regex=True)
    df["review"] = df["review"].str.strip()

    before = len(df)
    df.dropna(subset=["review"], inplace=True)
    df.drop_duplicates(subset=["review"], inplace=True)
    logger.info(f"Dropped {before - len(df)} null/duplicate rows")

    df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
    df.to_csv(clean_path, index=False)
    logger.info(f"Cleaned data saved — {len(df)} rows")


def validate_data(**context):
    clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    df = pd.read_csv(clean_path)

    required_cols = {"review", "label", "sentiment", "source"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"
    assert df["review"].isnull().sum() == 0,    "Null reviews found"
    assert df["sentiment"].isnull().sum() == 0, "Null sentiments found"
    assert len(df) >= 2000, f"Too few rows: {len(df)}"

    dist = df["sentiment"].value_counts(normalize=True)
    logger.info(f"Label distribution:\n{dist}")
    logger.info("Validation passed")


def compute_baseline_stats(**context):
    clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    df = pd.read_csv(clean_path)
    df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))

    stats = {
        "mean_review_length":  round(df["review_length"].mean(), 2),
        "std_review_length":   round(df["review_length"].std(), 2),
        "min_review_length":   int(df["review_length"].min()),
        "max_review_length":   int(df["review_length"].max()),
        "total_rows":          len(df),
        "label_distribution":  df["sentiment"].value_counts().to_dict(),
        "computed_at":         datetime.utcnow().isoformat(),
    }

    stats_path = os.path.join(PROCESSED_DIR, "baseline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Baseline stats: {stats}")


default_args = {
    "owner":            "review-pulse",
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="review_ingestion_pipeline",
    description="Ingest, clean, validate, and baseline UCI review data",
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

    t1 >> t2 >> t3 >> t4