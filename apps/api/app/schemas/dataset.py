from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    data: list[dict[str, Any]]


class DatasetRead(BaseModel):
    id: int
    project_id: int
    filename: str
    row_count: int
    column_count: int
    data: list[dict[str, Any]]
    created_at: datetime


class DatasetProfile(BaseModel):
    missing_values: dict[str, int]
    column_types: dict[str, str]


class DatasetUploadPreview(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    preview: list[dict[str, Any]]
    data: list[dict[str, Any]]
    profile: DatasetProfile


class DatasetUploadResult(BaseModel):
    dataset: DatasetRead
    preview: DatasetUploadPreview
