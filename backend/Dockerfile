FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV MLFLOW_URI=http://mlflow:5000
ENV MODEL_NAME=sentiment-tfidf-lr
ENV MODEL_STAGE=Production
ENV PROCESSED_DIR=/opt/airflow/data/processed

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]