from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger   # ← our rotating file logger

# Get the root logger — this writes to both stdout AND /opt/logs/app.log
logger = get_logger("review_pulse")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Sentiment Analysis API for Review Intelligence Platform",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Prometheus metrics — exposed at /metrics ─────────────────────────────────
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
).instrument(app).expose(app, endpoint="/metrics")

# ── CORS — allow all origins for local dev ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ── Startup: attempt to pre-load the model ───────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Try to load the model when the server starts.
    If no model is registered yet (training hasn't run), we log a warning
    and continue — the health endpoint will report 'degraded' until
    training completes and registers a model in MLflow.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"MLflow URI: {settings.MLFLOW_URI}")
    logger.info(f"Model: {settings.MODEL_NAME} / stage: {settings.MODEL_STAGE}")
    try:
        from backend.app.core.model_loader import load_model
        load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Model not loaded on startup (run training to fix): {e}")


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "app":     settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }