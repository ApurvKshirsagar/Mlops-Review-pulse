FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.0 \
    mlflow==3.11.1 \
    scikit-learn==1.8.0 \
    pandas==2.3.3 \
    numpy \
    pydantic==2.13.3 \
    python-multipart \
    prometheus-fastapi-instrumentator==7.0.0

COPY backend/ ./backend/

ENV PYTHONPATH=/app
ENV MLFLOW_URI=http://mlflow:5000
ENV MODEL_NAME=sentiment-tfidf-lr
ENV MODEL_STAGE=Production
ENV PROCESSED_DIR=/opt/airflow/data/processed

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]