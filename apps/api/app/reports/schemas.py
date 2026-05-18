from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    report_type: str
    title: str
    content_markdown: str
    source_summary: dict[str, Any]
    created_at: datetime


class ReportListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    report_type: str
    title: str
    created_at: datetime

