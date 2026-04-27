# Loads the trained model at startup and caches it as a module-level singleton.
#
# Loading strategy (tries in order):
#   1. MLflow registry — latest version of MODEL_NAME (MLflow v3 compatible)
#   2. Local pickle fallback — for running outside Docker without MLflow

import os
import pickle
import traceback

import mlflow.pyfunc
import mlflow.sklearn

from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger

logger = get_logger(__name__)

# Module-level singleton — loaded once at startup, reused for every request.
# This avoids the overhead of loading the model on every prediction call.
_model = None


def load_model():
    """
    Load the model from MLflow registry or local pickle fallback.
    Caches the result in _model so subsequent calls return immediately.
    """
    global _model
    if _model is not None:
        return _model

    mlflow.set_tracking_uri(settings.MLFLOW_URI)

    # ── Strategy 1: MLflow registry (latest version) ─────────────────────────
    # MLflow v3 dropped stage-based loading (models:/name/Production).
    # We fetch the latest registered version by version number instead.
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=settings.MLFLOW_URI)

        # get_latest_versions returns versions sorted ascending — [-1] is newest
        versions = client.get_latest_versions(settings.MODEL_NAME)
        if versions:
            latest = versions[-1]
            model_uri = f"models:/{settings.MODEL_NAME}/{latest.version}"
            _model = mlflow.sklearn.load_model(model_uri)
            logger.info(
                f"Loaded model from MLflow registry: {settings.MODEL_NAME} "
                f"v{latest.version} (run_id={latest.run_id})"
            )
            return _model
        else:
            logger.warning(f"No versions found for model '{settings.MODEL_NAME}' in MLflow registry")

    except Exception:
        logger.warning(
            f"MLflow registry load failed — will try local pickle:\n"
            + traceback.format_exc()
        )

    # ── Strategy 2: local pickle (fallback) ──────────────────────────────────
    # Used when running outside Docker or when MLflow is unavailable.
    # The pickle is saved by train.py alongside the MLflow artifact.
    local_path = os.path.join(
        os.environ.get("MODEL_DIR", "mlflow/artifacts/models"),
        "tfidf_lr_pipeline.pkl"
    )
    try:
        with open(local_path, "rb") as f:
            _model = pickle.load(f)
        logger.info(f"Loaded model from local pickle: {local_path}")
        return _model
    except Exception:
        logger.error(
            f"Local pickle load also failed ({local_path}):\n"
            + traceback.format_exc()
        )
        raise RuntimeError(
            "No model available. Run: docker-compose exec backend python -m backend.app.services.train"
        )


def get_model():
    """Return the cached model, loading it if not yet loaded."""
    return load_model()


def reload_model():
    """
    Force reload the model from MLflow (clears the cache).
    Call this after training completes to pick up the new version without restart.
    """
    global _model
    _model = None
    logger.info("Model cache cleared — reloading from MLflow...")
    return load_model()