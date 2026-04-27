import json
import logging
import os
import pickle
import subprocess
import time
import traceback

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.app.core.logging_config import get_logger

# Load .env for local runs (no-op inside Docker where vars are injected)
load_dotenv()

logger = get_logger(__name__)

# ── Config — all values come from environment / .env ─────────────────────────
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "airflow/data/processed")
MODEL_DIR     = os.environ.get("MODEL_DIR",     "mlflow/artifacts/models")
MLFLOW_URI    = os.environ.get("MLFLOW_URI",    "http://localhost:5000")
LOG_DIR       = os.environ.get("LOG_DIR",       "/opt/logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: write one JSON line to training.log after every run
# This gives a permanent, human-readable audit trail of every training event.
# ─────────────────────────────────────────────────────────────────────────────
def _log_training_event(model_name: str, run_id: str, metrics: dict) -> None:
    """Append one JSON line to LOG_DIR/training.log."""
    entry = {
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model":      model_name,
        "run_id":     run_id,
        "accuracy":   metrics.get("accuracy"),
        "f1_macro":   metrics.get("f1_macro"),
        "precision":  metrics.get("precision"),
        "recall":     metrics.get("recall"),
        "train_size": metrics.get("train_size"),
        "test_size":  metrics.get("test_size"),
    }
    log_path = os.path.join(LOG_DIR, "training.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(
        f"Training event logged — model={model_name} "
        f"accuracy={metrics.get('accuracy')} f1={metrics.get('f1_macro')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: create a Git tag so the code version is tied to the model version
# Non-fatal — if git is unavailable (CI, some Docker images) we just warn.
# ─────────────────────────────────────────────────────────────────────────────
def _create_git_tag(model_name: str, version: str, metrics: dict) -> None:
    """Create and push an annotated Git tag after model promotion."""
    tag = f"model/{model_name}/v{version}"
    message = (
        f"Model {model_name} v{version} promoted to Production. "
        f"accuracy={metrics.get('accuracy')} f1={metrics.get('f1_macro')}"
    )
    try:
        subprocess.run(["git", "tag", "-a", tag, "-m", message], check=True)
        subprocess.run(["git", "push", "origin", tag], check=True)
        logger.info(f"Git tag created and pushed: {tag}")
    except subprocess.CalledProcessError as exc:
        # Non-fatal: git may not be configured in all environments
        logger.warning(f"Git tagging skipped (non-fatal): {exc}")
    except FileNotFoundError:
        logger.warning("git binary not found — skipping tag creation")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load the cleaned CSV produced by the Airflow pipeline."""
    path = os.path.join(PROCESSED_DIR, "cleaned.csv")
    try:
        df = pd.read_csv(path)
        # Drop rows with missing review text or sentiment label
        df = df.dropna(subset=["review", "sentiment"])
        logger.info(f"Loaded {len(df)} rows. Distribution:\n{df['sentiment'].value_counts()}")
        return df
    except FileNotFoundError:
        logger.error(f"cleaned.csv not found at {path} — run the Airflow pipeline first")
        raise
    except Exception:
        logger.error("Unexpected error loading data:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: TF-IDF + Logistic Regression (fast baseline)
# ─────────────────────────────────────────────────────────────────────────────
def train_tfidf_lr(df: pd.DataFrame):
    """
    Train a TF-IDF vectoriser + Logistic Regression pipeline.
    Returns (run_id, metrics_dict) on success, raises on failure.
    """
    logger.info("Starting TF-IDF + Logistic Regression training...")

    X = df["review"].astype(str)
    y = df["sentiment"]

    # 80/20 stratified split to preserve class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sentiment-analysis")

    try:
        with mlflow.start_run(run_name="tfidf-logistic-regression"):

            # ── Hyperparameters ───────────────────────────────────────────────
            tfidf_params = {
                "max_features": 10000,   # top 10k vocabulary terms
                "ngram_range":  (1, 2),  # unigrams + bigrams
                "min_df":       2,       # ignore terms that appear only once
            }
            lr_params = {
                "C":        1.0,     # regularisation strength (lower = more regularised)
                "max_iter": 1000,    # enough iterations for convergence
                "solver":   "lbfgs",
            }

            # ── Build and train sklearn pipeline ─────────────────────────────
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                ("lr",    LogisticRegression(**lr_params)),
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # ── Compute metrics ───────────────────────────────────────────────
            metrics = {
                "accuracy":   round(accuracy_score(y_test, y_pred), 4),
                "f1_macro":   round(f1_score(y_test, y_pred, average="macro"), 4),
                "precision":  round(precision_score(y_test, y_pred, average="macro"), 4),
                "recall":     round(recall_score(y_test, y_pred, average="macro"), 4),
                "test_size":  len(X_test),
                "train_size": len(X_train),
            }
            logger.info(f"TF-IDF + LR metrics: {metrics}")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            # ── Log params and metrics to MLflow ─────────────────────────────
            mlflow.log_params({**tfidf_params, **lr_params, "ngram_range": "1,2"})
            mlflow.log_metrics(metrics)

            # Save and log classification report as a JSON artifact
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = os.path.join(MODEL_DIR, "tfidf_lr_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)

            # Register model in MLflow Model Registry
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="tfidf-lr-model",
                registered_model_name="sentiment-tfidf-lr",
            )

            # Also save a local pickle as fallback for model_loader.py
            model_path = os.path.join(MODEL_DIR, "tfidf_lr_pipeline.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)

            run_id = mlflow.active_run().info.run_id
            logger.info(f"TF-IDF LR run completed — run_id={run_id}")

            # Write to training.log for permanent audit trail
            _log_training_event("sentiment-tfidf-lr", run_id, metrics)

            return run_id, metrics

    except Exception:
        logger.error("TF-IDF LR training failed:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: DistilBERT (deep learning, optional)
# ─────────────────────────────────────────────────────────────────────────────
def train_distilbert(df: pd.DataFrame):
    """
    Fine-tune DistilBERT for binary sentiment classification.
    Returns (run_id, metrics_dict) or (None, None) if torch/transformers
    are not installed (e.g. in a CPU-only backend container).
    """
    try:
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            Trainer,
            TrainingArguments,
        )
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        logger.error("transformers/torch not installed — skipping DistilBERT training")
        return None, None

    logger.info("Starting DistilBERT training...")

    X = df["review"].astype(str).tolist()
    y = df["sentiment"].tolist()

    # Map string labels to integer class indices
    label2id = {"Negative": 0, "Positive": 1}
    id2label  = {0: "Negative", 1: "Positive"}
    y_encoded = [label2id[label] for label in y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Minimal torch Dataset wrapper required by HuggingFace Trainer
    class ReviewDataset(Dataset):
        def __init__(self, texts, labels):
            # Tokenise all texts up-front; truncate to 128 tokens max
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

    # Load pre-trained DistilBERT with a fresh classification head (2 classes)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sentiment-analysis")

    try:
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
                report_to="none",   # disable wandb/tensorboard
            )

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=-1)
                return {
                    "accuracy": accuracy_score(labels, preds),
                    "f1_macro": f1_score(labels, preds, average="macro"),
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
                "accuracy":   round(eval_results.get("eval_accuracy", 0), 4),
                "f1_macro":   round(eval_results.get("eval_f1_macro", 0), 4),
                "test_size":  len(X_test),
                "train_size": len(X_train),
            }
            logger.info(f"DistilBERT metrics: {metrics}")

            mlflow.log_params({
                "model":                  "distilbert-base-uncased",
                "num_train_epochs":       2,
                "per_device_train_batch": 16,
                "max_length":             128,
            })
            mlflow.log_metrics(metrics)
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                name="distilbert-model",
                registered_model_name="sentiment-distilbert",
                extra_pip_requirements=["torchvision"],
            )

            run_id = mlflow.active_run().info.run_id
            logger.info(f"DistilBERT run completed — run_id={run_id}")

            _log_training_event("sentiment-distilbert", run_id, metrics)

            return run_id, metrics

    except Exception:
        logger.error("DistilBERT training failed:\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison and registry promotion
# ─────────────────────────────────────────────────────────────────────────────
def compare_and_register_best(lr_metrics, bert_metrics, lr_run_id, bert_run_id) -> str:
    """
    Compare TF-IDF LR vs DistilBERT by F1-macro score.
    Promotes the winner to the MLflow 'Production' stage.
    Creates a Git tag so the exact code version is linked to the model.
    Returns the name of the best model.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW_URI)

    lr_f1   = lr_metrics["f1_macro"]
    bert_f1 = bert_metrics["f1_macro"] if bert_metrics else 0

    if bert_f1 > lr_f1:
        best_model   = "sentiment-distilbert"
        best_run     = bert_run_id
        best_metrics = bert_metrics
        logger.info(f"DistilBERT wins: F1={bert_f1:.4f} vs LR F1={lr_f1:.4f}")
    else:
        best_model   = "sentiment-tfidf-lr"
        best_run     = lr_run_id
        best_metrics = lr_metrics
        logger.info(f"TF-IDF LR wins: F1={lr_f1:.4f} vs DistilBERT F1={bert_f1:.4f}")

    # get_latest_versions returns a list sorted by version number ascending.
    # We take [-1] to get the most recently registered version of the best model.
    versions = client.get_latest_versions(best_model)
    promoted_version = None
    if versions:
        promoted_version = versions[-1].version
        client.transition_model_version_stage(
            name=best_model,
            version=promoted_version,
            stage="Production",
        )
        logger.info(f"Promoted {best_model} v{promoted_version} to Production")

        # Create a Git tag so this code commit is linked to the promoted model
        _create_git_tag(best_model, str(promoted_version), best_metrics)

    # Save a JSON summary for reference / debugging
    summary = {
        "tfidf_lr":   {"run_id": lr_run_id,   "metrics": lr_metrics},
        "distilbert": {"run_id": bert_run_id,  "metrics": bert_metrics},
        "best_model": best_model,
        "best_run_id": best_run,
    }
    summary_path = os.path.join(MODEL_DIR, "model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Comparison summary saved to {summary_path}")

    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        df = load_data()

        logger.info("=" * 60)
        logger.info("Step 1/2 — Training TF-IDF + Logistic Regression")
        logger.info("=" * 60)
        lr_run_id, lr_metrics = train_tfidf_lr(df)

        logger.info("=" * 60)
        logger.info("Step 2/2 — Training DistilBERT")
        logger.info("=" * 60)
        bert_run_id, bert_metrics = train_distilbert(df)

        if lr_metrics:
            compare_and_register_best(lr_metrics, bert_metrics, lr_run_id, bert_run_id)

        logger.info("Training pipeline complete!")

    except Exception:
        logger.error("Training pipeline failed:\n" + traceback.format_exc())
        raise