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
    message: str | None = None
    tool_used: bool = False
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None
    plan_executed: bool = False
    steps_summary: list[dict[str, Any]] | None = None
    tools_used: list[str] | None = None
    pending_action: dict[str, Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.message is None:
            self.message = self.reply


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
