from typing import Any
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelTrainingRequest(BaseModel):
    data: list[dict[str, Any]] | None = None
    target_column: str = Field(min_length=1)
    project_id: int | None = None
    dataset_id: int | None = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelTrainingResponse(BaseModel):
    model_run_id: int | None = None
    project_id: int | None = None
    dataset_id: int | None = None
    problem_type: str
    task_type: str | None = None
    model_type: str
    target_column: str | None = None
    metrics: dict[str, float]
    feature_importance: list[FeatureImportance]


class ModelRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    dataset_id: int | None
    target_column: str
    task_type: str
    model_type: str
    metrics: dict[str, float]
    feature_importance: list[FeatureImportance]
    created_at: datetime
