import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
import json
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from sklearn.pipeline import Pipeline
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "airflow/data/processed")
MODEL_DIR     = os.environ.get("MODEL_DIR",     "mlflow/artifacts/models")
MLFLOW_URI    = os.environ.get("MLFLOW_URI",    "http://localhost:5000")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    df = pd.read_csv(path)
    # Drop neutral-ish rows with missing sentiment
    df = df.dropna(subset=["review", "sentiment"])
    logger.info(f"Loaded {len(df)} rows. Distribution:\n{df['sentiment'].value_counts()}")
    return df

def train_tfidf_lr(df):
    """Train TF-IDF + Logistic Regression baseline."""

    X = df["review"].astype(str)
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sentiment-analysis")

    with mlflow.start_run(run_name="tfidf-logistic-regression"):

        # ── Hyperparameters ──
        tfidf_params = {
            "max_features": 10000,
            "ngram_range":  (1, 2),
            "min_df":       2,
        }
        lr_params = {
            "C":           1.0,
            "max_iter":    1000,
            "solver":      "lbfgs",
        }

        # ── Build pipeline ──
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("lr",    LogisticRegression(**lr_params)),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # ── Metrics ──
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "f1_macro":  round(f1_score(y_test, y_pred, average="macro"), 4),
            "precision": round(precision_score(y_test, y_pred, average="macro"), 4),
            "recall":    round(recall_score(y_test, y_pred, average="macro"), 4),
            "test_size": len(X_test),
            "train_size": len(X_train),
        }

        logger.info(f"TF-IDF + LR Metrics: {metrics}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        # ── Log to MLflow ──
        mlflow.log_params({**tfidf_params, **lr_params, "ngram_range": "1,2"})
        mlflow.log_metrics(metrics)

        # Log classification report as artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = os.path.join(MODEL_DIR, "tfidf_lr_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="tfidf-lr-model",
            registered_model_name="sentiment-tfidf-lr",
        )

        # Save locally too
        model_path = os.path.join(MODEL_DIR, "tfidf_lr_pipeline.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"TF-IDF LR run_id: {run_id}")
        return run_id, metrics


def train_distilbert(df):
    """Train DistilBERT sentiment classifier."""
    try:
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            Trainer, TrainingArguments
        )
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        logger.error("transformers/torch not installed. Skipping DistilBERT.")
        return None, None

    X = df["review"].astype(str).tolist()
    y = df["sentiment"].tolist()

    # Encode labels
    label2id = {"Negative": 0, "Positive": 1}
    id2label = {0: "Negative", 1: "Positive"}
    y_encoded = [label2id[label] for label in y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = ReviewDataset(X_train, y_train)
    test_dataset  = ReviewDataset(X_test,  y_test)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sentiment-analysis")

    with mlflow.start_run(run_name="distilbert-classifier"):

        training_args = TrainingArguments(
            output_dir=os.path.join(MODEL_DIR, "distilbert"),
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
            report_to="none",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy":  accuracy_score(labels, preds),
                "f1_macro":  f1_score(labels, preds, average="macro"),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        metrics = {
            "accuracy":  round(eval_results.get("eval_accuracy", 0), 4),
            "f1_macro":  round(eval_results.get("eval_f1_macro", 0), 4),
            "test_size": len(X_test),
            "train_size": len(X_train),
        }

        mlflow.log_params({
            "model":                   "distilbert-base-uncased",
            "num_train_epochs":        2,
            "per_device_train_batch":  16,
            "max_length":              128,
        })
        mlflow.log_metrics(metrics)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            name="distilbert-model",
            registered_model_name="sentiment-distilbert",
            extra_pip_requirements=["torchvision"],
        )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"DistilBERT run_id: {run_id}")
        return run_id, metrics


def compare_and_register_best(lr_metrics, bert_metrics, lr_run_id, bert_run_id):
    """Compare both models and promote the better one to Production."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW_URI)

    lr_f1   = lr_metrics["f1_macro"]
    bert_f1 = bert_metrics["f1_macro"] if bert_metrics else 0

    if bert_f1 > lr_f1:
        best_model = "sentiment-distilbert"
        best_run   = bert_run_id
        logger.info(f"DistilBERT wins: F1={bert_f1} vs LR F1={lr_f1}")
    else:
        best_model = "sentiment-tfidf-lr"
        best_run   = lr_run_id
        logger.info(f"TF-IDF LR wins: F1={lr_f1} vs DistilBERT F1={bert_f1}")

    # Get latest version and promote to Production
    versions = client.get_latest_versions(best_model)
    if versions:
        client.transition_model_version_stage(
            name=best_model,
            version=versions[-1].version,
            stage="Production",
        )
        logger.info(f"Promoted {best_model} v{versions[-1].version} to Production")

    # Save comparison summary
    summary = {
        "tfidf_lr":    {"run_id": lr_run_id,   "metrics": lr_metrics},
        "distilbert":  {"run_id": bert_run_id,  "metrics": bert_metrics},
        "best_model":  best_model,
        "best_run_id": best_run,
    }
    summary_path = os.path.join(MODEL_DIR, "model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Comparison saved to {summary_path}")
    return best_model


if __name__ == "__main__":
    df = load_data()

    logger.info("=" * 50)
    logger.info("Training TF-IDF + Logistic Regression...")
    logger.info("=" * 50)
    lr_run_id, lr_metrics = train_tfidf_lr(df)

    logger.info("=" * 50)
    logger.info("Training DistilBERT...")
    logger.info("=" * 50)
    bert_run_id, bert_metrics = train_distilbert(df)

    if lr_metrics:
        compare_and_register_best(lr_metrics, bert_metrics, lr_run_id, bert_run_id)

    logger.info("Training complete!")