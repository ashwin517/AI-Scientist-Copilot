from typing import Any

from pydantic import BaseModel, Field

from app.agent.tool_registry import ToolRegistry


PLAN_TOOL_NAME = "execution_plan"
PLAN_MIN_STEPS = 3
PLAN_MAX_STEPS = 8


class PlanStep(BaseModel):
    step_id: int
    tool_name: str
    arguments_json: dict[str, Any] = Field(default_factory=dict)
    purpose: str
    status: str = "pending"
    result_summary: str | None = None


def create_execution_plan(
    user_query: str,
    project_context: dict[str, Any] | None,
    registry: ToolRegistry | None = None,
) -> list[PlanStep]:
    active_registry = registry or ToolRegistry()
    normalized = _normalize(user_query)
    if not _is_supported_multi_step_intent(normalized):
        return []

    steps = [
        PlanStep(
            step_id=1,
            tool_name="list_project_memory",
            purpose="Inspect saved project memory and current workspace facts.",
        ),
        PlanStep(
            step_id=2,
            tool_name="get_dataset_summary",
            arguments_json={"latest": True},
            purpose="Inspect the latest project dataset when one is available.",
        ),
        PlanStep(
            step_id=3,
            tool_name="answer_document_question",
            arguments_json={
                "question": _document_summary_question(user_query),
                "top_k": 5,
            },
            purpose="Inspect uploaded document context through the existing RAG tool.",
        ),
        PlanStep(
            step_id=4,
            tool_name="explain_latest_model",
            purpose="Inspect the latest saved model result without training a new model.",
        ),
        PlanStep(
            step_id=5,
            tool_name="explain_latest_simulation",
            purpose="Inspect the latest saved simulation without running a new simulation.",
        ),
        PlanStep(
            step_id=6,
            tool_name="explain_latest_optimization",
            arguments_json={"include_latest_simulation_comparison": True},
            purpose=(
                "Inspect the latest saved optimization and compare it with the "
                "latest simulation when possible."
            ),
        ),
    ]

    if _should_include_experiment_recommendations(normalized, project_context):
        steps.append(
            PlanStep(
                step_id=7,
                tool_name="recommend_next_experiment",
                arguments_json={"count": 3},
                purpose=(
                    "Recommend next simulated experiment candidates from an "
                    "existing optimization result."
                ),
            )
        )

    return _approved_bounded_steps(steps, active_registry)


def _approved_bounded_steps(
    steps: list[PlanStep],
    registry: ToolRegistry,
) -> list[PlanStep]:
    approved_steps = [step for step in steps if registry.get(step.tool_name) is not None]
    approved_steps = approved_steps[:PLAN_MAX_STEPS]
    if len(approved_steps) < PLAN_MIN_STEPS:
        return []
    for index, step in enumerate(approved_steps, start=1):
        step.step_id = index
    return approved_steps


def _is_supported_multi_step_intent(message: str) -> bool:
    phrases = (
        "analyze my project",
        "analyse my project",
        "analyze this project",
        "analyse this project",
        "analyze the project",
        "analyse the project",
        "suggest next experiments",
        "suggest next experiment",
        "review my project status",
        "review project status",
        "review the project status",
        "what should i do next",
    )
    return any(phrase in message for phrase in phrases)


def _should_include_experiment_recommendations(
    message: str,
    project_context: dict[str, Any] | None,
) -> bool:
    if any(token in message for token in ("experiment", "experiments", "do next")):
        return True
    if not project_context:
        return False
    return bool(project_context.get("latest_optimization_run_id"))


def _document_summary_question(user_query: str) -> str:
    query = user_query.strip()
    if not query:
        return "Summarize uploaded project documents for project status and next experiments."
    return (
        "Summarize uploaded project documents that are relevant to this project "
        f"planning request: {query}"
    )


def _normalize(message: str) -> str:
    return " ".join(message.strip().casefold().split())
