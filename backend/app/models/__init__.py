from pydantic import BaseModel
from typing import List, Optional

class SinglePredictRequest(BaseModel):
    review: str

    model_config = {"json_schema_extra": {"example": {"review": "This product is amazing!"}}}


class SinglePredictResponse(BaseModel):
    review: str
    sentiment: str
    confidence: float
    probabilities: dict


class BatchPredictRequest(BaseModel):
    reviews: List[str]


class BatchPredictResponse(BaseModel):
    predictions: List[SinglePredictResponse]
    total:       int


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    version:     str