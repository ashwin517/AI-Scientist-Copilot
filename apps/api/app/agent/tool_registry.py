from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent.tools.dataset_tools import (
    get_dataset_summary,
    list_datasets,
    show_missing_values,
)
from app.agent.tools.document_tools import answer_document_question
from app.agent.tools.memory_tools import (
    delete_project_memory,
    list_project_memory,
    upsert_project_memory,
)
from app.agent.tools.model_tools import (
    explain_latest_model,
    list_model_runs,
    train_baseline_model,
)
from app.agent.tools.optimization_tools import (
    explain_latest_optimization,
    list_optimization_runs,
    optimize_batch_reactor,
    recommend_next_experiment,
)
from app.agent.tools.report_tools import (
    explain_latest_report,
    generate_project_report_tool,
    list_reports,
    review_latest_report,
)
from app.agent.tools.simulation_tools import (
    compare_simulation_runs,
    explain_latest_simulation,
    list_simulation_runs,
    run_batch_reactor_simulation,
)
from app.agent.tools.workflow_tools import (
    compare_workflow_runs,
    explain_latest_workflow,
    list_workflow_runs,
    run_project_analysis_workflow_tool,
)


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
            "explain_latest_model": ToolDefinition(
                name="explain_latest_model",
                handler=explain_latest_model,
                description="Explain the latest saved model run using persisted metrics and feature importance.",
            ),
            "answer_document_question": ToolDefinition(
                name="answer_document_question",
                handler=answer_document_question,
                description="Answer a question from uploaded project documents.",
            ),
            "list_project_memory": ToolDefinition(
                name="list_project_memory",
                handler=list_project_memory,
                description="List project-scoped memory records.",
            ),
            "upsert_project_memory": ToolDefinition(
                name="upsert_project_memory",
                handler=upsert_project_memory,
                description="Create or update a project-scoped memory record.",
            ),
            "delete_project_memory": ToolDefinition(
                name="delete_project_memory",
                handler=delete_project_memory,
                description="Delete a project-scoped memory record by key.",
            ),
            "run_batch_reactor_simulation": ToolDefinition(
                name="run_batch_reactor_simulation",
                handler=run_batch_reactor_simulation,
                description="Run and persist a simple A -> B -> C batch reactor simulation.",
            ),
            "list_simulation_runs": ToolDefinition(
                name="list_simulation_runs",
                handler=list_simulation_runs,
                description="List recent saved simulation runs in a project.",
            ),
            "explain_latest_simulation": ToolDefinition(
                name="explain_latest_simulation",
                handler=explain_latest_simulation,
                description="Explain the latest saved simulation run.",
            ),
            "compare_simulation_runs": ToolDefinition(
                name="compare_simulation_runs",
                handler=compare_simulation_runs,
                description="Compare two saved simulation runs, defaulting to the latest two.",
            ),
            "optimize_batch_reactor": ToolDefinition(
                name="optimize_batch_reactor",
                handler=optimize_batch_reactor,
                description="Search simple batch reactor operating conditions to improve yield while penalizing impurity.",
            ),
            "explain_latest_optimization": ToolDefinition(
                name="explain_latest_optimization",
                handler=explain_latest_optimization,
                description="Explain the latest saved optimization result and its yield/impurity tradeoff.",
            ),
            "list_optimization_runs": ToolDefinition(
                name="list_optimization_runs",
                handler=list_optimization_runs,
                description="List recent saved optimization runs in a project.",
            ),
            "recommend_next_experiment": ToolDefinition(
                name="recommend_next_experiment",
                handler=recommend_next_experiment,
                description="Recommend the next simulated batch reactor experiments from the latest optimization.",
            ),
            "run_project_analysis_workflow": ToolDefinition(
                name="run_project_analysis_workflow",
                handler=run_project_analysis_workflow_tool,
                description="Inspect existing project memory and assets, then summarize project status and next actions.",
            ),
            "list_workflow_runs": ToolDefinition(
                name="list_workflow_runs",
                handler=list_workflow_runs,
                description="List recent project workflow runs.",
            ),
            "explain_latest_workflow": ToolDefinition(
                name="explain_latest_workflow",
                handler=explain_latest_workflow,
                description="Explain the latest saved project workflow run and its recommendations.",
            ),
            "compare_workflow_runs": ToolDefinition(
                name="compare_workflow_runs",
                handler=compare_workflow_runs,
                description="Compare the latest two project workflow runs.",
            ),
            "generate_project_report": ToolDefinition(
                name="generate_project_report",
                handler=generate_project_report_tool,
                description="Generate and persist a deterministic markdown project report from current workspace state.",
            ),
            "list_reports": ToolDefinition(
                name="list_reports",
                handler=list_reports,
                description="List recent generated project reports.",
            ),
            "explain_latest_report": ToolDefinition(
                name="explain_latest_report",
                handler=explain_latest_report,
                description="Explain what the latest generated project report contains.",
            ),
            "review_latest_report": ToolDefinition(
                name="review_latest_report",
                handler=review_latest_report,
                description="Review the latest generated report and suggest improvements without modifying it.",
            ),
        }

    def get(self, tool_name: str) -> ToolDefinition | None:
        return self._tools.get(tool_name)

    def names(self) -> list[str]:
        return list(self._tools)
