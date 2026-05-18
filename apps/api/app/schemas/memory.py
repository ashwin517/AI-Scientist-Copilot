from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProjectMemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    memory_type: str
    key: str
    value: Any
    source: str | None
    created_at: datetime
    updated_at: datetime


class ProjectMemoryUpsert(BaseModel):
    key: str = Field(min_length=1, max_length=255)
    value: Any
    memory_type: str = Field(default="fact", min_length=1, max_length=64)
    source: str | None = Field(default=None, max_length=255)
