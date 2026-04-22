import logging
import numpy as np
from backend.app.core.model_loader import get_model

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "Negative", 1: "Positive"}
CONFIDENCE_THRESHOLD = 0.65  # below this → Neutral

def predict_single(review: str) -> dict:
    model = get_model()
    proba = model.predict_proba([review])[0]
    classes = model.classes_

    max_conf = float(np.max(proba))
    pred_class = classes[int(np.argmax(proba))]

    # Apply neutral threshold
    if max_conf < CONFIDENCE_THRESHOLD:
        sentiment = "Neutral"
    else:
        sentiment = pred_class

    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    return {
        "review":        review,
        "sentiment":     sentiment,
        "confidence":    round(max_conf, 4),
        "probabilities": prob_dict,
    }


def predict_batch(reviews: list) -> list:
    return [predict_single(r) for r in reviews]