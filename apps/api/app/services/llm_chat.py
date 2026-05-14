import json
from typing import Any

import requests

from app.schemas.chat import ChatResponse


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT_SECONDS = 60
MAX_PROMPT_CHARACTERS = 12000
MAX_CONTEXT_JSON_CHARACTERS = 4000
MAX_USER_MESSAGE_CHARACTERS = 2000
MAX_COLUMN_NAMES = 80
MAX_MISSING_VALUE_COLUMNS = 80
MAX_PREVIEW_ROWS = 5
MAX_PREVIEW_COLUMNS = 12
MAX_FEATURE_IMPORTANCE_ITEMS = 10
MAX_CELL_CHARACTERS = 120

OLLAMA_UNAVAILABLE_REPLY = (
    "I could not reach the local Ollama service. Please make sure Ollama is "
    "running on http://localhost:11434 and that the llama3.2:3b model is available."
)


def generate_chat_reply(
    message: str,
    dataset_summary: dict[str, Any] | None = None,
    model_result: dict[str, Any] | None = None,
) -> ChatResponse:
    prompt = _build_prompt(message, dataset_summary, model_result)

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
    except requests.RequestException:
        return ChatResponse(reply=OLLAMA_UNAVAILABLE_REPLY)

    try:
        data = response.json()
    except ValueError:
        return ChatResponse(
            reply=(
                "Ollama responded, but the response was not valid JSON. "
                "Please try again."
            )
        )

    reply = data.get("response")
    if not isinstance(reply, str) or not reply.strip():
        return ChatResponse(
            reply=(
                "Ollama responded, but I could not find a usable assistant reply. "
                "Please try again."
            )
        )

    return ChatResponse(reply=reply.strip())


def _build_prompt(
    message: str,
    dataset_summary: dict[str, Any] | None,
    model_result: dict[str, Any] | None,
) -> str:
    dataset_context = _summarize_dataset_context(dataset_summary)
    model_context = _summarize_model_context(model_result)
    user_message = _truncate_text(message.strip(), MAX_USER_MESSAGE_CHARACTERS)

    prompt = f"""System instruction:
You are AI Scientist Copilot, a scientific data analysis assistant.

Grounding rules:
- Reason only from the dataset and model context provided below.
- If the available context is insufficient, say what information is unavailable.
- Avoid overclaiming, unsupported causal claims, or invented dataset details.
- Explain results like a data scientist or process scientist.
- Suggest practical next analysis steps when appropriate.
- Do not assume access to raw data beyond the summarized context.

Dataset context:
{_to_bounded_json(dataset_context)}

Model context:
{_to_bounded_json(model_context)}

User question:
{user_message}

Assistant response:"""

    return _truncate_text(prompt, MAX_PROMPT_CHARACTERS)


def _summarize_dataset_context(
    dataset_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if dataset_summary is None:
        return {
            "available": False,
            "note": "No dataset context was provided.",
        }

    column_names = _as_list(dataset_summary.get("column_names"))
    limited_column_names = column_names[:MAX_COLUMN_NAMES]
    profile = _as_dict(dataset_summary.get("profile"))
    missing_values = _as_dict(profile.get("missing_values"))

    summary: dict[str, Any] = {
        "available": True,
        "filename": dataset_summary.get("filename"),
        "rows": dataset_summary.get("rows"),
        "columns": dataset_summary.get("columns"),
        "column_names": limited_column_names,
        "missing_values": dict(
            list(missing_values.items())[:MAX_MISSING_VALUE_COLUMNS]
        ),
        "raw_dataset_included": False,
    }

    if len(column_names) > MAX_COLUMN_NAMES:
        summary["column_names_omitted"] = len(column_names) - MAX_COLUMN_NAMES

    if len(missing_values) > MAX_MISSING_VALUE_COLUMNS:
        summary["missing_value_columns_omitted"] = (
            len(missing_values) - MAX_MISSING_VALUE_COLUMNS
        )

    preview_rows = _build_limited_preview(
        _as_list(dataset_summary.get("preview")),
        limited_column_names,
    )
    if preview_rows:
        summary["preview_rows"] = preview_rows
        summary["preview_note"] = (
            "Only the small preview rows supplied by the frontend are included; "
            "the full raw dataset is not included."
        )
    else:
        summary["preview_note"] = (
            "Preview rows were unavailable or too large to include safely."
        )

    return summary


def _summarize_model_context(
    model_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if model_result is None:
        return {
            "available": False,
            "note": "No model result context was provided.",
        }

    feature_importance = _as_list(model_result.get("feature_importance"))
    top_feature_importance = _build_top_feature_importance(feature_importance)

    summary: dict[str, Any] = {
        "available": True,
        "problem_type": model_result.get("problem_type"),
        "model_type": model_result.get("model_type"),
        "metrics": _as_dict(model_result.get("metrics")),
        "top_feature_importance": top_feature_importance,
    }

    if len(feature_importance) > MAX_FEATURE_IMPORTANCE_ITEMS:
        summary["feature_importance_items_omitted"] = (
            len(feature_importance) - MAX_FEATURE_IMPORTANCE_ITEMS
        )

    return summary


def _build_limited_preview(
    preview_rows: list[Any],
    column_names: list[Any],
) -> list[dict[str, Any]]:
    if len(preview_rows) > MAX_PREVIEW_ROWS:
        preview_rows = preview_rows[:MAX_PREVIEW_ROWS]

    limited_columns = [str(column) for column in column_names[:MAX_PREVIEW_COLUMNS]]
    limited_preview: list[dict[str, Any]] = []

    for row in preview_rows:
        if not isinstance(row, dict):
            continue

        if limited_columns:
            row_items = [
                (column, row.get(column))
                for column in limited_columns
                if column in row
            ]
        else:
            row_items = list(row.items())[:MAX_PREVIEW_COLUMNS]

        limited_preview.append(
            {
                str(key): _sanitize_context_value(value)
                for key, value in row_items
            }
        )

    return limited_preview


def _build_top_feature_importance(
    feature_importance: list[Any],
) -> list[dict[str, Any]]:
    top_items: list[dict[str, Any]] = []

    for item in feature_importance[:MAX_FEATURE_IMPORTANCE_ITEMS]:
        if not isinstance(item, dict):
            continue

        top_items.append(
            {
                "feature": _sanitize_context_value(item.get("feature")),
                "importance": item.get("importance"),
            }
        )

    return top_items


def _sanitize_context_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, MAX_CELL_CHARACTERS)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_text(str(value), MAX_CELL_CHARACTERS)


def _to_bounded_json(value: dict[str, Any]) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, default=str)
    return _truncate_text(text, MAX_CONTEXT_JSON_CHARACTERS)


def _truncate_text(value: str, max_characters: int) -> str:
    if len(value) <= max_characters:
        return value
    omitted = len(value) - max_characters
    return value[:max_characters].rstrip() + f"\n...[truncated {omitted} characters]"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []
