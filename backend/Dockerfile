# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory for all subsequent commands
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source code ──────────────────────────────────────────────
COPY backend/ ./backend/

# ── Runtime environment defaults ─────────────────────────────────────────────
# All of these are overridden by docker-compose env_file / environment blocks.
# They serve as documentation of what the app expects.
ENV PYTHONPATH=/app
ENV MLFLOW_URI=http://mlflow:5000
ENV MODEL_NAME=sentiment-tfidf-lr
ENV MODEL_STAGE=Production
ENV PROCESSED_DIR=/opt/airflow/data/processed
ENV LOG_DIR=/opt/logs

# Expose the port uvicorn listens on
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]