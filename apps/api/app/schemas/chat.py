from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    project_id: int | None = None
    dataset_summary: dict[str, Any] | None = None
    model_result: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    reply: str


class ChatMessageCreate(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
