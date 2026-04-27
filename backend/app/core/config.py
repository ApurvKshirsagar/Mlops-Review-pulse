from dotenv import load_dotenv
import os

# Load .env from the project root (one level up from wherever Python runs).
# Does nothing if .env is absent (e.g. inside Docker where vars are injected).
load_dotenv()


class Settings:
    # MLflow tracking server URL
    MLFLOW_URI: str     = os.environ.get("MLFLOW_URI",     "http://mlflow:5000")

    # Registered model name and stage to serve
    MODEL_NAME: str     = os.environ.get("MODEL_NAME",     "sentiment-tfidf-lr")
    MODEL_STAGE: str    = os.environ.get("MODEL_STAGE",    "Production")

    # Path to processed CSVs (baseline_stats.json lives here too)
    PROCESSED_DIR: str  = os.environ.get("PROCESSED_DIR",  "/opt/airflow/data/processed")

    # Host-mounted log directory — logs are written here, NOT inside the image
    LOG_DIR: str        = os.environ.get("LOG_DIR",        "/opt/logs")

    # App metadata
    APP_VERSION: str    = os.environ.get("APP_VERSION",    "1.0.0")
    APP_NAME: str       = os.environ.get("APP_NAME",       "Review Pulse API")

    # Drift detection settings (used by Airflow DAG)
    DRIFT_THRESHOLD: float = float(os.environ.get("DRIFT_THRESHOLD", "0.1"))
    RETRAIN_ON_DRIFT: bool = os.environ.get("RETRAIN_ON_DRIFT", "true").lower() == "true"


settings = Settings()