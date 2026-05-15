from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent.tools.dataset_tools import (
    get_dataset_summary,
    list_datasets,
    show_missing_values,
)
from app.agent.tools.document_tools import answer_document_question
from app.agent.tools.model_tools import list_model_runs, train_baseline_model


ToolCallable = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: ToolCallable
    description: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {
            "list_datasets": ToolDefinition(
                name="list_datasets",
                handler=list_datasets,
                description="List datasets saved in a project.",
            ),
            "get_dataset_summary": ToolDefinition(
                name="get_dataset_summary",
                handler=get_dataset_summary,
                description="Summarize a saved dataset.",
            ),
            "show_missing_values": ToolDefinition(
                name="show_missing_values",
                handler=show_missing_values,
                description="Show missing value counts for a saved dataset.",
            ),
            "train_baseline_model": ToolDefinition(
                name="train_baseline_model",
                handler=train_baseline_model,
                description="Train the existing baseline model on a saved dataset.",
            ),
            "list_model_runs": ToolDefinition(
                name="list_model_runs",
                handler=list_model_runs,
                description="List previously trained model runs when available.",
            ),
            "answer_document_question": ToolDefinition(
                name="answer_document_question",
                handler=answer_document_question,
                description="Answer a question from uploaded project documents.",
            ),
        }

    def get(self, tool_name: str) -> ToolDefinition | None:
        return self._tools.get(tool_name)

    def names(self) -> list[str]:
        return list(self._tools)
