from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class WorkflowStepRead(BaseModel):
    name: str
    status: str
    summary: str
    data: dict[str, Any] | None = None


class WorkflowRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    workflow_type: str
    status: str
    steps: list[WorkflowStepRead]
    result: dict[str, Any]
    created_at: datetime
    completed_at: datetime | None

