from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import re
import os
import logging
import json

# Configure module logger for Airflow task logs
logger = logging.getLogger(__name__)

# Base directories mounted from Docker volumes
RAW_DIR = "/opt/airflow/data/raw"
PROCESSED_DIR = "/opt/airflow/data/processed"

# Task 1: Read raw datasets from multiple sources and combine them
def ingest_data(**context):
    files = {
        "amazon": os.path.join(RAW_DIR, "sentiment labelled sentences/amazon_cells_labelled.txt"),
        "imdb": os.path.join(RAW_DIR, "sentiment labelled sentences/imdb_labelled.txt"),
        "yelp": os.path.join(RAW_DIR, "sentiment labelled sentences/yelp_labelled.txt"),
    }

    frames = []
    for source, path in files.items():
        # Load tab-separated file with columns: review, label
        df = pd.read_csv(path, sep="\t", header=None, names=["review", "label"])
        df["source"] = source  # Track source platform
        frames.append(df)
        logger.info(f"Loaded {len(df)} rows from {source}")

    # Merge all sources into one dataframe
    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_path = os.path.join(PROCESSED_DIR, "raw_combined.csv")
    combined.to_csv(raw_path, index=False)

    logger.info(f"Ingestion complete — {len(combined)} total rows")

    # Store row count in XCom for downstream visibility
    context["ti"].xcom_push(key="row_count", value=len(combined))

# Task 2: Clean text data and prepare labels
def clean_data(**context):
    raw_path = os.path.join(PROCESSED_DIR, "raw_combined.csv")
    clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")

    df = pd.read_csv(raw_path)

    # Standard text normalization
    df["review"] = df["review"].str.lower()
    df["review"] = df["review"].str.replace(r"<.*?>", "", regex=True)  # Remove HTML tags
    df["review"] = df["review"].str.replace(r"[^a-z0-9\s]", "", regex=True)  # Remove punctuation/symbols
    df["review"] = df["review"].str.strip()

    before = len(df)

    # Remove invalid or duplicate rows
    df.dropna(subset=["review"], inplace=True)
    df.drop_duplicates(subset=["review"], inplace=True)

    logger.info(f"Dropped {before - len(df)} null/duplicate rows")

    # Human-readable sentiment labels
    df["sentiment"] = df["label"].map({0: "Negative", 1: "Positive"})
    df.to_csv(clean_path, index=False)

    logger.info(f"Cleaned data saved — {len(df)} rows")

# Task 3: Validate schema and quality checks
def validate_data(**context):
    clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    df = pd.read_csv(clean_path)

    required_cols = {"review", "label", "sentiment", "source"}

    # Basic assertions for data quality
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"
    assert df["review"].isnull().sum() == 0, "Null reviews found"
    assert df["sentiment"].isnull().sum() == 0, "Null sentiments found"
    assert len(df) >= 2000, f"Too few rows: {len(df)}"

    dist = df["sentiment"].value_counts(normalize=True)
    logger.info(f"Label distribution:\n{dist}")
    logger.info("Validation passed")

# Task 4: Compute baseline metrics for monitoring/drift checks
def compute_baseline_stats(**context):
    clean_path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    df = pd.read_csv(clean_path)

    # Number of words per review
    df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))

    stats = {
        "mean_review_length": round(df["review_length"].mean(), 2),
        "std_review_length": round(df["review_length"].std(), 2),
        "min_review_length": int(df["review_length"].min()),
        "max_review_length": int(df["review_length"].max()),
        "total_rows": len(df),
        "label_distribution": df["sentiment"].value_counts().to_dict(),
        "computed_at": datetime.utcnow().isoformat(),
    }

    stats_path = os.path.join(PROCESSED_DIR, "baseline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Baseline stats: {stats}")

# Default behavior for all tasks
default_args = {
    "owner": "review-pulse",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

# Define DAG metadata and schedule
with DAG(
    dag_id="review_ingestion_pipeline",
    description="Ingest, clean, validate, and baseline UCI review data",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["nlp", "ingestion", "review-pulse"],
) as dag:

    # Create Airflow tasks mapped to Python functions
    t1 = PythonOperator(task_id="ingest_data", python_callable=ingest_data, provide_context=True)
    t2 = PythonOperator(task_id="clean_data", python_callable=clean_data, provide_context=True)
    t3 = PythonOperator(task_id="validate_data", python_callable=validate_data, provide_context=True)
    t4 = PythonOperator(task_id="compute_baseline_stats", python_callable=compute_baseline_stats, provide_context=True)

    # Execution order / dependencies
    t1 >> t2 >> t3 >> t4
