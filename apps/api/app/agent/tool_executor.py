import inspect

from sqlalchemy.orm import Session

from app.agent.tool_registry import ToolRegistry
from app.agent.tool_schemas import ToolIntent, ToolResult


class ToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def execute(
        self,
        db: Session,
        project_id: int | None,
        intent: ToolIntent,
        project_memory: dict[str, object] | None = None,
    ) -> ToolResult:
        if not intent.requires_tool or not intent.tool_name:
            return ToolResult(
                tool_name=intent.tool_name or "none",
                success=False,
                error="No tool was requested.",
            )

        tool = self.registry.get(intent.tool_name)
        if tool is None:
            return ToolResult(
                tool_name=intent.tool_name,
                success=False,
                error="Unknown or unapproved tool requested.",
            )

        if project_id is None:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                error="A project_id is required to use project tools.",
            )

        try:
            arguments = dict(intent.arguments)
            if _accepts_project_memory(tool.handler):
                arguments["project_memory"] = project_memory or {}
            data = tool.handler(db=db, project_id=project_id, **arguments)
        except TypeError as exc:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                error=f"Invalid tool arguments: {exc}",
            )
        except ValueError as exc:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                error=str(exc),
            )

        return ToolResult(tool_name=tool.name, success=True, data=data)


def _accepts_project_memory(handler: object) -> bool:
    return "project_memory" in inspect.signature(handler).parameters
