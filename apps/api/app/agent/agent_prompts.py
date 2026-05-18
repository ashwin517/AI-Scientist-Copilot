import json
from typing import Any

from app.agent.tool_schemas import ToolResult


MAX_MEMORY_CONTEXT_JSON_CHARACTERS = 2500


def build_tool_summary_prompt(
    user_message: str,
    tool_result: ToolResult,
    project_memory: dict[str, Any] | None = None,
) -> str:
    memory_context = project_memory or {
        "available": False,
        "note": "No project memory context was provided.",
    }
    return f"""System instruction:
You are AI Scientist Copilot. Summarize the approved tool result below for a scientist.

Rules:
- Do not claim the tool did anything outside the provided result.
- Mention limitations or errors plainly.
- Do not describe internal tool execution mechanics.
- If a dataset list is empty, explain that this active project has no saved datasets
  and that the user should upload a CSV dataset first.
- Keep the answer concise and actionable.

Project memory:
{_to_bounded_json(memory_context)}

User message:
{user_message}

Tool name:
{tool_result.tool_name}

Tool success:
{tool_result.success}

Tool result:
{tool_result.model_dump_json(indent=2)}

Assistant response:"""


def _to_bounded_json(value: dict[str, Any]) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, default=str)
    if len(text) <= MAX_MEMORY_CONTEXT_JSON_CHARACTERS:
        return text
    omitted = len(text) - MAX_MEMORY_CONTEXT_JSON_CHARACTERS
    return (
        text[:MAX_MEMORY_CONTEXT_JSON_CHARACTERS].rstrip()
        + f"\n...[truncated {omitted} characters]"
    )
