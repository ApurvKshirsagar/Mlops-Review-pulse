import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import router
from backend.app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Sentiment Analysis API for Review Intelligence Platform",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        from backend.app.core.model_loader import load_model
        load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Model not loaded on startup: {e}")

@app.get("/")
def root():
    return {
        "app":     settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
    }