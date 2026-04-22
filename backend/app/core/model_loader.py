import mlflow.sklearn
import mlflow.pyfunc
import pickle
import logging
import os
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    # Try MLflow registry first
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_URI)
        model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
        _model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from MLflow registry: {model_uri}")
        return _model
    except Exception as e:
        logger.warning(f"MLflow registry load failed: {e}")

    # Fallback to local pickle
    try:
        local_path = os.path.join("mlflow", "artifacts", "models", "tfidf_lr_pipeline.pkl")
        with open(local_path, "rb") as f:
            _model = pickle.load(f)
        logger.info(f"Loaded model from local pickle: {local_path}")
        return _model
    except Exception as e:
        logger.error(f"Local model load failed: {e}")
        raise RuntimeError("No model available. Run training first.")

def get_model():
    return load_model()