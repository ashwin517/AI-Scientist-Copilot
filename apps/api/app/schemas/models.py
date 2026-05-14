from typing import Any

from pydantic import BaseModel, Field


class ModelTrainingRequest(BaseModel):
    data: list[dict[str, Any]]
    target_column: str = Field(min_length=1)


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelTrainingResponse(BaseModel):
    problem_type: str
    model_type: str
    metrics: dict[str, float]
    feature_importance: list[FeatureImportance]
