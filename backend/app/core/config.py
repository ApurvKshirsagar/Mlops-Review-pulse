import os

class Settings:
    MLFLOW_URI: str     = os.environ.get("MLFLOW_URI",     "http://mlflow:5000")
    MODEL_NAME: str     = os.environ.get("MODEL_NAME",     "sentiment-tfidf-lr")
    MODEL_STAGE: str    = os.environ.get("MODEL_STAGE",    "Production")
    PROCESSED_DIR: str  = os.environ.get("PROCESSED_DIR",  "/opt/airflow/data/processed")
    APP_VERSION: str    = "1.0.0"
    APP_NAME: str       = "Review Pulse API"

settings = Settings()