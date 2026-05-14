from app.agent.tool_schemas import ToolResult


def build_tool_summary_prompt(user_message: str, tool_result: ToolResult) -> str:
    return f"""System instruction:
You are AI Scientist Copilot. Summarize the approved tool result below for a scientist.

Rules:
- Do not claim the tool did anything outside the provided result.
- Mention limitations or errors plainly.
- Do not describe internal tool execution mechanics.
- If a dataset list is empty, explain that this active project has no saved datasets
  and that upload previews must be saved to the project first.
- Keep the answer concise and actionable.

User message:
{user_message}

Tool name:
{tool_result.tool_name}

Tool success:
{tool_result.success}

Tool result:
{tool_result.model_dump_json(indent=2)}

Assistant response:"""
