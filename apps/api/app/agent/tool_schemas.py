from typing import Any

from pydantic import BaseModel, Field


class ToolIntent(BaseModel):
    requires_tool: bool
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str | None = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    error: str | None = None

