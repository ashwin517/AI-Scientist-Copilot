import json
from typing import Any

import requests
from sqlalchemy.orm import Session

from app.agent.agent_prompts import build_tool_summary_prompt
from app.agent.intent_parser import IntentParser
from app.agent.pending_action import (
    clear_pending_action,
    get_pending_action,
    pending_arguments,
    pending_missing_fields,
    save_pending_action,
)
from app.agent.slot_filling import fill_target_column
from app.agent.tool_executor import ToolExecutor
from app.agent.tool_schemas import ToolIntent
from app.agent.tool_schemas import ToolResult
from app.agent.tools.dataset_tools import resolve_dataset
from app.agent.tools.model_tools import _available_columns
from app.schemas.chat import ChatResponse
from app.schemas.chat import ChatMessageCreate
from app.services.chat_messages import create_chat_message
from app.services.llm_chat import (
    DEFAULT_MODEL,
    OLLAMA_GENERATE_URL,
    OLLAMA_UNAVAILABLE_REPLY,
    REQUEST_TIMEOUT_SECONDS,
    generate_chat_reply,
)
from app.services.projects import get_project


class AgentService:
    def __init__(
        self,
        parser: IntentParser | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        self.parser = parser or IntentParser()
        self.executor = executor or ToolExecutor()

    def handle_message(
        self,
        db: Session,
        project_id: int | None,
        message: str,
        dataset_summary: dict[str, Any] | None = None,
        model_result: dict[str, Any] | None = None,
    ) -> ChatResponse:
        if project_id is not None and get_project(db, project_id) is None:
            return ChatResponse(reply="Project not found.")

        if project_id is not None:
            create_chat_message(
                db,
                project_id,
                ChatMessageCreate(role="user", content=message),
            )

        if project_id is not None:
            pending_response = self._handle_pending_action(db, project_id, message)
            if pending_response is not None:
                self._save_assistant_message(db, project_id, pending_response.reply)
                return pending_response

        intent = self.parser.parse(message)
        if not intent.requires_tool:
            response = generate_chat_reply(
                message=message,
                dataset_summary=dataset_summary,
                model_result=model_result,
            )
            self._save_assistant_message(db, project_id, response.reply)
            return response

        if (
            project_id is not None
            and intent.tool_name == "train_baseline_model"
            and not intent.arguments.get("target_column")
        ):
            response = self._create_pending_train_action(
                db,
                project_id,
                intent.arguments,
            )
            self._save_assistant_message(db, project_id, response.reply)
            return response

        result = self.executor.execute(db=db, project_id=project_id, intent=intent)
        response = ChatResponse(reply=self._summarize_tool_result(message, result))
        self._save_assistant_message(db, project_id, response.reply)
        return response

    def _handle_pending_action(
        self,
        db: Session,
        project_id: int,
        message: str,
    ) -> ChatResponse | None:
        pending_action = get_pending_action(db, project_id)
        if pending_action is None:
            return None

        if _is_cancel_message(message):
            clear_pending_action(db, project_id)
            return ChatResponse(reply="Canceled the pending action.")

        if pending_action.tool_name != "train_baseline_model":
            clear_pending_action(db, project_id)
            return None

        arguments = pending_arguments(pending_action)
        missing_fields = pending_missing_fields(pending_action)
        if "target_column" not in missing_fields:
            clear_pending_action(db, project_id)
            return None

        try:
            available_columns = _available_columns_for_training(
                db,
                project_id,
                arguments,
            )
        except ValueError as exc:
            clear_pending_action(db, project_id)
            return ChatResponse(reply=str(exc))

        slot_result = fill_target_column(message, available_columns)
        if slot_result.value is None:
            if slot_result.ambiguous:
                reply = (
                    "That column name is ambiguous. Available columns are: "
                    f"{_format_columns(available_columns)}"
                )
            else:
                reply = slot_result.error or (
                    "Which target column should I use? Available columns: "
                    f"{_format_columns(available_columns)}"
                )
            return ChatResponse(reply=reply)

        arguments["target_column"] = slot_result.value
        result = self.executor.execute(
            db=db,
            project_id=project_id,
            intent=ToolIntent(
                requires_tool=True,
                tool_name=pending_action.tool_name,
                arguments=arguments,
                confidence=1.0,
                explanation="Filled pending action target_column.",
            ),
        )
        if result.success:
            clear_pending_action(db, project_id)

        return ChatResponse(reply=self._summarize_tool_result(message, result))

    def _create_pending_train_action(
        self,
        db: Session,
        project_id: int,
        arguments: dict[str, Any],
    ) -> ChatResponse:
        try:
            available_columns = _available_columns_for_training(
                db,
                project_id,
                arguments,
            )
        except ValueError as exc:
            return ChatResponse(reply=str(exc))

        save_pending_action(
            db,
            project_id,
            "train_baseline_model",
            arguments,
            ["target_column"],
        )
        return ChatResponse(
            reply=(
                "Which target column should I use? "
                f"Available columns: {_format_columns(available_columns)}"
            )
        )

    def _summarize_tool_result(self, message: str, result: ToolResult) -> str:
        if not result.success:
            return _deterministic_summary(result)

        if result.tool_name == "list_datasets" and result.data.get("count") == 0:
            return _deterministic_summary(result)

        prompt = build_tool_summary_prompt(message, result)
        try:
            response = requests.post(
                OLLAMA_GENERATE_URL,
                json={
                    "model": DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return _deterministic_summary(result)

        reply = data.get("response")
        if not isinstance(reply, str) or not reply.strip():
            return _deterministic_summary(result)
        if reply.strip() == OLLAMA_UNAVAILABLE_REPLY:
            return _deterministic_summary(result)
        return reply.strip()

    def _save_assistant_message(
        self,
        db: Session,
        project_id: int | None,
        reply: str,
    ) -> None:
        if project_id is None:
            return
        create_chat_message(
            db,
            project_id,
            ChatMessageCreate(role="assistant", content=reply),
        )


def _deterministic_summary(result: ToolResult) -> str:
    if not result.success:
        return result.error or "The requested tool could not run."

    if result.tool_name == "list_datasets":
        datasets = result.data.get("datasets", [])
        if not datasets:
            project_id = result.data.get("project_id")
            return (
                f"No datasets are saved in active project {project_id} yet. "
                "If you uploaded a CSV in the workspace, make sure it was saved "
                "to this project, or switch back to the project where it was saved."
            )
        lines = ["Saved datasets in this project:"]
        for dataset in datasets:
            lines.append(
                "- {filename} (id {id}): {row_count} rows, {column_count} columns".format(
                    **dataset
                )
            )
        return "\n".join(lines)

    if result.tool_name == "get_dataset_summary":
        dataset = result.data.get("dataset", {})
        columns = result.data.get("column_names", [])
        return (
            f"{dataset.get('filename', 'Dataset')} has "
            f"{dataset.get('row_count')} rows and {dataset.get('column_count')} columns. "
            f"Columns: {', '.join(columns[:20])}."
        )

    if result.tool_name == "show_missing_values":
        dataset = result.data.get("dataset", {})
        total = result.data.get("total_missing_values", 0)
        columns = result.data.get("columns_with_missing", {})
        if not columns:
            return f"{dataset.get('filename', 'The dataset')} has no missing values."
        details = ", ".join(f"{column}: {count}" for column, count in columns.items())
        return (
            f"{dataset.get('filename', 'The dataset')} has {total} missing values. "
            f"Columns with missing values: {details}."
        )

    if result.tool_name == "train_baseline_model":
        model_result = result.data.get("model_result", {})
        return (
            f"I trained a {model_result.get('model_type')} "
            f"{model_result.get('problem_type')} model to predict "
            f"{result.data.get('target_column')}. The run was saved to your "
            f"workspace as model run #{result.data.get('model_run_id')}."
        )

    if result.tool_name == "list_model_runs":
        model_runs = result.data.get("model_runs", [])
        if not model_runs:
            return "No model runs are saved in this project yet."
        lines = ["Saved model runs in this project:"]
        for model_run in model_runs[:10]:
            lines.append(
                "- #{id}: {model_type} {task_type} predicting {target_column}".format(
                    **model_run
                )
            )
        return "\n".join(lines)

    return json.dumps(result.data, indent=2, sort_keys=True, default=str)


def _available_columns_for_training(
    db: Session,
    project_id: int,
    arguments: dict[str, Any],
) -> list[str]:
    try:
        dataset = resolve_dataset(
            db,
            project_id,
            arguments.get("dataset_id"),
            bool(arguments.get("latest", True)),
        )
    except ValueError as exc:
        if str(exc) == "No dataset is available for this project.":
            raise ValueError(
                "This project has no saved datasets yet. Please upload a CSV dataset first."
            ) from exc
        raise

    try:
        rows = json.loads(dataset.raw_data_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Saved dataset could not be decoded.") from exc

    if not isinstance(rows, list):
        raise ValueError("Saved dataset has an invalid format.")

    return _available_columns(rows)


def _format_columns(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "(none)"


def _is_cancel_message(message: str) -> bool:
    return message.strip().casefold() in {
        "cancel",
        "stop",
        "never mind",
        "nevermind",
    }
