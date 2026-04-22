import logging
import pandas as pd
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.models import (
    SinglePredictRequest, SinglePredictResponse,
    BatchPredictRequest,  BatchPredictResponse,
    HealthResponse,
)
from backend.app.services.predictor import predict_single, predict_batch
from backend.app.core.model_loader import get_model
from backend.app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=settings.APP_VERSION,
    )


@router.get("/ready", tags=["System"])
def readiness_check():
    try:
        get_model()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict", response_model=SinglePredictResponse, tags=["Prediction"])
def predict(request: SinglePredictRequest):
    """Predict sentiment for a single review."""
    try:
        result = predict_single(request.review)
        return SinglePredictResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch_endpoint(request: BatchPredictRequest):
    """Predict sentiment for a list of reviews."""
    if len(request.reviews) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 reviews per request")
    try:
        results = predict_batch(request.reviews)
        return BatchPredictResponse(
            predictions=[SinglePredictResponse(**r) for r in results],
            total=len(results),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-csv", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """Upload a CSV with a 'review' column and get sentiment predictions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if "review" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have a 'review' column")

        reviews = df["review"].fillna("").astype(str).tolist()
        results = predict_batch(reviews)

        return BatchPredictResponse(
            predictions=[SinglePredictResponse(**r) for r in results],
            total=len(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", tags=["Analytics"])
def get_stats():
    """Return baseline stats from the data pipeline."""
    import json, os
    stats_path = os.path.join(
        settings.PROCESSED_DIR, "baseline_stats.json"
    )
    try:
        with open(stats_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Stats not available yet")