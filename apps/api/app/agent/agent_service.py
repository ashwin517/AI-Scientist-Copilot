import json
from typing import Any

import requests
from sqlalchemy.orm import Session

from app.agent.agent_prompts import build_tool_summary_prompt
from app.agent.intent_parser import IntentParser
from app.agent.planner import PLAN_TOOL_NAME
from app.agent.planner import PlanStep
from app.agent.planner import create_execution_plan
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
from app.services.memory_service import (
    build_memory_prompt_context,
    list_memory,
    memory_records_to_dict,
    update_project_summary,
    upsert_memory,
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
            return _chat_response("Project not found.")

        if project_id is not None:
            create_chat_message(
                db,
                project_id,
                ChatMessageCreate(role="user", content=message),
            )

        if project_id is not None:
            update_project_summary(db, project_id)
        memory_records = list_memory(db, project_id) if project_id is not None else []
        memory_values = memory_records_to_dict(memory_records)
        memory_context = build_memory_prompt_context(memory_records)

        if project_id is not None:
            pending_response = self._handle_pending_action(
                db,
                project_id,
                message,
                memory_values,
                memory_context,
            )
            if pending_response is not None:
                self._save_assistant_message(db, project_id, pending_response.reply)
                return pending_response

        if project_id is not None:
            plan = create_execution_plan(
                message,
                memory_values,
                registry=self.executor.registry,
            )
            if plan:
                response = self._execute_plan(
                    db,
                    project_id,
                    message,
                    plan,
                    memory_values,
                )
                self._save_assistant_message(db, project_id, response.reply)
                return response

        intent = self.parser.parse(message, project_memory=memory_values)
        if not intent.requires_tool:
            response = generate_chat_reply(
                message=message,
                dataset_summary=dataset_summary,
                model_result=model_result,
                project_memory=memory_context,
            )
            self._save_assistant_message(db, project_id, response.reply)
            return response

        if (
            project_id is not None
            and intent.tool_name == "train_baseline_model"
            and not intent.arguments.get("target_column")
            and not memory_values.get("selected_target_column")
        ):
            response = self._create_pending_train_action(
                db,
                project_id,
                intent.arguments,
                memory_values,
            )
            self._save_assistant_message(db, project_id, response.reply)
            return response

        result = self.executor.execute(
            db=db,
            project_id=project_id,
            intent=intent,
            project_memory=memory_values,
        )
        response = self._tool_response(
            message,
            result,
            project_memory=memory_context,
        )
        self._save_assistant_message(db, project_id, response.reply)
        return response

    def _handle_pending_action(
        self,
        db: Session,
        project_id: int,
        message: str,
        memory_values: dict[str, Any],
        memory_context: dict[str, Any],
    ) -> ChatResponse | None:
        pending_action = get_pending_action(db, project_id)
        if pending_action is None:
            return None

        if _is_cancel_message(message):
            clear_pending_action(db, project_id)
            return _chat_response("Canceled the pending action.")

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
                memory_values,
            )
        except ValueError as exc:
            clear_pending_action(db, project_id)
            return _chat_response(str(exc))

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
            return _chat_response(
                reply,
                pending_action=_pending_action_payload(
                    pending_action.tool_name,
                    arguments,
                    missing_fields,
                ),
            )

        arguments["target_column"] = slot_result.value
        upsert_memory(
            db,
            project_id,
            "selected_target_column",
            slot_result.value,
            memory_type="user_decision",
            source="target_column_confirmation",
        )
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
            project_memory=memory_values,
        )
        if result.success:
            clear_pending_action(db, project_id)

        return self._tool_response(
            message,
            result,
            project_memory=memory_context,
        )

    def _execute_plan(
        self,
        db: Session,
        project_id: int,
        message: str,
        plan: list[PlanStep],
        memory_values: dict[str, Any],
    ) -> ChatResponse:
        results: list[ToolResult] = []
        for step in plan:
            result = self.executor.execute(
                db=db,
                project_id=project_id,
                intent=ToolIntent(
                    requires_tool=True,
                    tool_name=step.tool_name,
                    arguments=step.arguments_json,
                    confidence=1.0,
                    explanation=f"Execution plan step {step.step_id}: {step.purpose}",
                ),
                project_memory=memory_values,
            )
            results.append(result)
            step.status = "completed" if result.success else "failed"
            step.result_summary = _bounded_text(_deterministic_summary(result))

        reply = _synthesize_plan_response(message, plan, results)
        steps_summary = [_plan_step_payload(step) for step in plan]
        tools_used = [step.tool_name for step in plan]
        return _chat_response(
            reply,
            tool_used=True,
            tool_name=PLAN_TOOL_NAME,
            tool_result={
                "tool_name": PLAN_TOOL_NAME,
                "success": True,
                "plan_executed": True,
                "steps_summary": steps_summary,
                "tools_used": tools_used,
                "completed_steps": sum(1 for result in results if result.success),
                "failed_steps": sum(1 for result in results if not result.success),
            },
            plan_executed=True,
            steps_summary=steps_summary,
            tools_used=tools_used,
        )

    def _create_pending_train_action(
        self,
        db: Session,
        project_id: int,
        arguments: dict[str, Any],
        memory_values: dict[str, Any],
    ) -> ChatResponse:
        try:
            available_columns = _available_columns_for_training(
                db,
                project_id,
                arguments,
                memory_values,
            )
        except ValueError as exc:
            return _chat_response(str(exc))

        save_pending_action(
            db,
            project_id,
            "train_baseline_model",
            arguments,
            ["target_column"],
        )
        reply = (
            "Which target column should I use? "
            f"Available columns: {_format_columns(available_columns)}"
        )
        return _chat_response(
            reply,
            pending_action=_pending_action_payload(
                "train_baseline_model",
                arguments,
                ["target_column"],
            ),
        )

    def _tool_response(
        self,
        message: str,
        result: ToolResult,
        project_memory: dict[str, Any] | None = None,
    ) -> ChatResponse:
        reply = self._summarize_tool_result(message, result, project_memory)
        return _chat_response(
            reply,
            tool_used=True,
            tool_name=result.tool_name,
            tool_result=_structured_tool_result(result),
        )

    def _summarize_tool_result(
        self,
        message: str,
        result: ToolResult,
        project_memory: dict[str, Any] | None = None,
    ) -> str:
        if not result.success:
            return _deterministic_summary(result)

        if result.tool_name == "list_datasets" and result.data.get("count") == 0:
            return _deterministic_summary(result)

        if result.tool_name == "train_baseline_model":
            return _deterministic_summary(result)

        if result.tool_name == "explain_latest_model":
            return _deterministic_summary(result)

        if result.tool_name == "answer_document_question":
            return _deterministic_summary(result)

        if result.tool_name in {
            "list_project_memory",
            "upsert_project_memory",
            "delete_project_memory",
            "run_batch_reactor_simulation",
            "list_simulation_runs",
            "explain_latest_simulation",
            "compare_simulation_runs",
            "optimize_batch_reactor",
            "explain_latest_optimization",
            "list_optimization_runs",
            "recommend_next_experiment",
            "run_project_analysis_workflow",
            "list_workflow_runs",
            "explain_latest_workflow",
            "compare_workflow_runs",
            "generate_project_report",
            "list_reports",
            "explain_latest_report",
            "review_latest_report",
        }:
            return _deterministic_summary(result)

        prompt = build_tool_summary_prompt(
            message,
            result,
            project_memory=project_memory,
        )
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
                "Please upload a CSV dataset in this project, or switch back to "
                "the project where it was uploaded."
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
        notes = _memory_notes_text(result)
        return (
            f"{notes}"
            f"{dataset.get('filename', 'Dataset')} has "
            f"{dataset.get('row_count')} rows and {dataset.get('column_count')} columns. "
            f"Columns: {', '.join(columns[:20])}."
        )

    if result.tool_name == "show_missing_values":
        dataset = result.data.get("dataset", {})
        total = result.data.get("total_missing_values", 0)
        columns = result.data.get("columns_with_missing", {})
        notes = _memory_notes_text(result)
        if not columns:
            return (
                f"{notes}{dataset.get('filename', 'The dataset')} "
                "has no missing values."
            )
        details = ", ".join(f"{column}: {count}" for column, count in columns.items())
        return (
            f"{notes}"
            f"{dataset.get('filename', 'The dataset')} has {total} missing values. "
            f"Columns with missing values: {details}."
        )

    if result.tool_name == "train_baseline_model":
        model_result = result.data.get("model_result", {})
        notes = _memory_notes_text(result)
        return (
            f"{notes}"
            f"I trained a {model_result.get('model_type')} "
            f"{model_result.get('problem_type')} model to predict "
            f"{result.data.get('target_column')}. The run was saved to your "
            f"workspace as model run #{result.data.get('model_run_id')}."
        )

    if result.tool_name == "list_model_runs":
        model_runs = result.data.get("model_runs", [])
        if not model_runs:
            return "No model runs are saved in this project yet."
        notes = _memory_notes_text(result)
        lines = ["Saved model runs in this project:"]
        for model_run in model_runs[:10]:
            lines.append(
                "- #{id}: {model_type} {task_type} predicting {target_column}".format(
                    **model_run
                )
            )
        return notes + "\n".join(lines)

    if result.tool_name == "explain_latest_model":
        if not result.data.get("model_available", False):
            return str(result.data.get("message") or "No trained model is available yet.")

        dataset = result.data.get("dataset") or {}
        dataset_text = ""
        if isinstance(dataset, dict) and dataset.get("filename"):
            dataset_text = f" on dataset {dataset.get('filename')}"
        metrics = result.data.get("metrics", {})
        features = result.data.get("top_features", [])
        limitations = result.data.get("limitations", [])
        next_steps = result.data.get("suggested_next_steps", [])

        lines = [
            (
                f"Latest model run #{result.data.get('model_run_id')} is a "
                f"{result.data.get('model_type')} {result.data.get('task_type')} "
                f"model predicting {result.data.get('target_column')}{dataset_text}."
            )
        ]
        if isinstance(metrics, dict) and metrics:
            metric_text = ", ".join(
                f"{key}: {value}" for key, value in metrics.items()
            )
            lines.append(f"Metrics: {metric_text}.")
        if isinstance(features, list) and features:
            feature_text = ", ".join(
                str(feature.get("feature"))
                for feature in features[:5]
                if isinstance(feature, dict) and feature.get("feature")
            )
            if feature_text:
                lines.append(f"Top predictive features: {feature_text}.")
        if limitations:
            lines.append(str(limitations[0]))
        if isinstance(next_steps, list) and next_steps:
            lines.append("Suggested next steps: " + " ".join(str(step) for step in next_steps[:3]))
        return "\n".join(lines)

    if result.tool_name == "answer_document_question":
        return str(result.data.get("answer") or "")

    if result.tool_name == "list_project_memory":
        memories = result.data.get("memories", [])
        return _format_project_memory(memories)

    if result.tool_name == "upsert_project_memory":
        if not result.data.get("updated", False):
            available_columns = result.data.get("available_columns", [])
            if available_columns:
                return (
                    f"{result.data.get('error')} Available columns are: "
                    f"{_format_columns([str(column) for column in available_columns])}."
                )
            return str(result.data.get("error") or "I could not update project memory.")

        key = result.data.get("key")
        value = result.data.get("value")
        if key == "selected_target_column":
            matched_dataset = result.data.get("matched_dataset")
            if isinstance(matched_dataset, dict) and matched_dataset.get("filename"):
                return (
                    f"I will use {value} as the target column from now on, "
                    f"using dataset {matched_dataset.get('filename')}."
                )
            return f"I will use {value} as the target column from now on."
        return f"I will remember that for this project: {value}."

    if result.tool_name == "delete_project_memory":
        if result.data.get("needs_clarification"):
            return str(result.data.get("message"))
        key = result.data.get("key")
        if result.data.get("deleted"):
            return f"I forgot {key} for this project."
        return f"I did not find saved project memory for {key}."

    if result.tool_name == "run_batch_reactor_simulation":
        simulation_input = result.data.get("input", {})
        return (
            f"Ran batch reactor simulation #{result.data.get('simulation_run_id')} "
            f"at {simulation_input.get('temperature')} C for "
            f"{simulation_input.get('batch_time')} minutes. "
            f"Final yield: {result.data.get('final_yield')}; "
            f"impurity: {result.data.get('final_impurity')}; "
            f"conversion: {result.data.get('conversion')}. "
            "This is a simple benchmark model, not real chemistry calibration."
        )

    if result.tool_name == "list_simulation_runs":
        simulation_runs = result.data.get("simulation_runs", [])
        if not simulation_runs:
            return "No simulation runs are saved in this project yet."
        lines = ["Recent simulation runs:"]
        for simulation_run in simulation_runs[:10]:
            input_values = simulation_run.get("input", {})
            result_values = simulation_run.get("result", {})
            lines.append(
                "- #{id}: {simulation_type}, {temperature} C for {batch_time} min; "
                "yield {final_yield}, impurity {final_impurity}, conversion {conversion}".format(
                    id=simulation_run.get("simulation_run_id"),
                    simulation_type=simulation_run.get("simulation_type"),
                    temperature=input_values.get("temperature"),
                    batch_time=input_values.get("batch_time"),
                    final_yield=result_values.get("final_yield"),
                    final_impurity=result_values.get("final_impurity"),
                    conversion=result_values.get("conversion"),
                )
            )
        lines.append("These are simplified simulated reactor runs, not calibrated chemistry.")
        return "\n".join(lines)

    if result.tool_name == "explain_latest_simulation":
        if not result.data.get("simulation_available", False):
            return str(result.data.get("message") or "No simulation runs are saved yet.")
        simulation_run_id = result.data.get("simulation_run_id")
        interpretation = result.data.get("interpretation")
        note = result.data.get("model_note")
        return (
            f"Latest simulation #{simulation_run_id}: {interpretation} "
            f"{note}"
        )

    if result.tool_name == "compare_simulation_runs":
        if not result.data.get("comparison_available", False):
            return str(result.data.get("message") or "No simulation comparison is available.")
        baseline = result.data.get("baseline", {})
        candidate = result.data.get("candidate", {})
        interpretation = result.data.get("interpretation")
        note = result.data.get("model_note")
        return (
            f"Compared simulation #{baseline.get('simulation_run_id')} with "
            f"#{candidate.get('simulation_run_id')}: {interpretation} {note}"
        )

    if result.tool_name == "optimize_batch_reactor":
        best_inputs = result.data.get("best_inputs", {})
        if not isinstance(best_inputs, dict):
            best_inputs = {}
        return (
            f"Optimization run #{result.data.get('optimization_run_id')} suggests "
            f"{best_inputs.get('temperature_c')} C, "
            f"{best_inputs.get('batch_time_min')} minutes, initial concentration "
            f"{best_inputs.get('initial_concentration')}, and catalyst factor "
            f"{best_inputs.get('catalyst_factor')}. "
            f"Best yield: {result.data.get('best_final_yield')}; impurity: "
            f"{result.data.get('best_final_impurity')}; conversion: "
            f"{result.data.get('best_conversion')}; objective: "
            f"{result.data.get('objective_value')}. "
            "This is a simplified grid-search result from the educational simulator, "
            "not a validated real-plant recommendation."
        )

    if result.tool_name == "explain_latest_optimization":
        if not result.data.get("optimization_available", False):
            return str(result.data.get("message") or "No optimization runs are saved yet.")
        comparison = result.data.get("simulation_comparison")
        comparison_text = ""
        if isinstance(comparison, dict):
            if comparison.get("comparison_available"):
                differences = comparison.get("result_differences", {})
                comparison_text = (
                    " Compared with the latest simulation, predicted differences "
                    f"are {differences}."
                )
            else:
                comparison_text = f" {comparison.get('message')}"
        return (
            f"Latest optimization #{result.data.get('optimization_run_id')}: "
            f"{result.data.get('explanation')} {result.data.get('model_note')}"
            f"{comparison_text}"
        )

    if result.tool_name == "list_optimization_runs":
        optimization_runs = result.data.get("optimization_runs", [])
        if not optimization_runs:
            return "No optimization runs are saved in this project yet."
        lines = ["Recent optimization runs:"]
        for optimization_run in optimization_runs[:10]:
            best_inputs = optimization_run.get("best_inputs", {})
            lines.append(
                "- #{id}: {optimization_type}; {temperature} C for {batch_time} min; "
                "yield {final_yield}, impurity {final_impurity}, objective {objective_value}".format(
                    id=optimization_run.get("optimization_run_id"),
                    optimization_type=optimization_run.get("optimization_type"),
                    temperature=best_inputs.get("temperature_c")
                    if isinstance(best_inputs, dict)
                    else "-",
                    batch_time=best_inputs.get("batch_time_min")
                    if isinstance(best_inputs, dict)
                    else "-",
                    final_yield=optimization_run.get("best_final_yield"),
                    final_impurity=optimization_run.get("best_final_impurity"),
                    objective_value=optimization_run.get("objective_value"),
                )
            )
        lines.append("These are simplified optimization runs, not plant-validated recipes.")
        return "\n".join(lines)

    if result.tool_name == "recommend_next_experiment":
        if not result.data.get("recommendation_available", False):
            return str(result.data.get("message") or "No recommendation is available.")
        recommendations = result.data.get("recommendations", [])
        lines = [
            "Recommended next simulated experiments from the latest optimization:"
        ]
        if isinstance(recommendations, list):
            for recommendation in recommendations[:3]:
                if not isinstance(recommendation, dict):
                    continue
                inputs = recommendation.get("inputs", {})
                if not isinstance(inputs, dict):
                    inputs = {}
                lines.append(
                    "- #{rank}: {temperature} C, {batch_time} min, initial {initial}, "
                    "catalyst {catalyst}; yield {yield_value}, impurity {impurity}. "
                    "{reason}".format(
                        rank=recommendation.get("rank"),
                        temperature=inputs.get("temperature_c"),
                        batch_time=inputs.get("batch_time_min"),
                        initial=inputs.get("initial_concentration"),
                        catalyst=inputs.get("catalyst_factor"),
                        yield_value=recommendation.get("final_yield"),
                        impurity=recommendation.get("final_impurity"),
                        reason=recommendation.get("reason"),
                    )
                )
        lines.append(str(result.data.get("note")))
        return "\n".join(lines)

    if result.tool_name == "run_project_analysis_workflow":
        workflow_result = result.data.get("result", {})
        if not isinstance(workflow_result, dict):
            return "I ran the project analysis workflow, but the result was not readable."

        assets = workflow_result.get("current_assets", {})
        gaps = workflow_result.get("gaps", [])
        actions = workflow_result.get("recommended_next_actions", [])
        lines = [
            f"Project status: {workflow_result.get('summary')}",
            "Current assets: " + _format_asset_status(assets),
        ]
        if isinstance(gaps, list) and gaps:
            lines.append("Gaps: " + "; ".join(str(gap) for gap in gaps[:5]) + ".")
        else:
            lines.append("Gaps: none obvious from saved project state.")
        if isinstance(actions, list) and actions:
            lines.append("Recommended next 3 actions:")
            for index, action in enumerate(actions[:3], start=1):
                lines.append(f"{index}. {action}")
        return "\n".join(lines)

    if result.tool_name == "list_workflow_runs":
        workflow_runs = result.data.get("workflow_runs", [])
        if not workflow_runs:
            return "No workflow runs exist for this project yet. Ask me to analyze this project to create the first project analysis."
        lines = ["Recent workflow runs:"]
        if isinstance(workflow_runs, list):
            for workflow_run in workflow_runs[:10]:
                if not isinstance(workflow_run, dict):
                    continue
                summary = workflow_run.get("summary") or "No summary saved."
                lines.append(
                    "- #{id}: {workflow_type}, {status}. {summary}".format(
                        id=workflow_run.get("workflow_run_id"),
                        workflow_type=workflow_run.get("workflow_type"),
                        status=workflow_run.get("status"),
                        summary=summary,
                    )
                )
        return "\n".join(lines)

    if result.tool_name == "explain_latest_workflow":
        if not result.data.get("workflow_available", False):
            return str(result.data.get("message") or "No workflow runs exist yet.")
        lines = [
            (
                f"Latest workflow #{result.data.get('workflow_run_id')} "
                f"({result.data.get('workflow_type')}) finished with status "
                f"{result.data.get('status')}."
            )
        ]
        summary = result.data.get("summary")
        if summary:
            lines.append(f"Summary: {summary}")
        gaps = result.data.get("gaps", [])
        if isinstance(gaps, list) and gaps:
            lines.append("Gaps: " + "; ".join(str(gap) for gap in gaps[:5]) + ".")
        actions = result.data.get("recommended_next_actions", [])
        if isinstance(actions, list) and actions:
            lines.append("Recommended next actions:")
            for index, action in enumerate(actions[:3], start=1):
                lines.append(f"{index}. {action}")
        return "\n".join(lines)

    if result.tool_name == "compare_workflow_runs":
        if not result.data.get("comparison_available", False):
            return str(result.data.get("message") or "No workflow comparison is available.")
        latest = result.data.get("latest_workflow", {})
        previous = result.data.get("previous_workflow", {})
        changes = result.data.get("major_project_changes", [])
        gaps = result.data.get("gaps", {})
        recommendations = result.data.get("recommendations", {})
        lines = [
            (
                f"Compared workflow #{previous.get('workflow_run_id')} to "
                f"#{latest.get('workflow_run_id')}."
            )
        ]
        if isinstance(changes, list) and changes:
            change_text = ", ".join(
                _format_asset_change(change)
                for change in changes[:5]
                if isinstance(change, dict)
            )
            if change_text:
                lines.append(f"Major project changes: {change_text}.")
        else:
            lines.append("Major project changes: no asset availability changes detected.")
        if isinstance(gaps, dict):
            new_gaps = gaps.get("new", [])
            resolved_gaps = gaps.get("resolved", [])
            if new_gaps:
                lines.append("New gaps: " + "; ".join(str(gap) for gap in new_gaps[:3]) + ".")
            if resolved_gaps:
                lines.append(
                    "Resolved gaps: "
                    + "; ".join(str(gap) for gap in resolved_gaps[:3])
                    + "."
                )
        if isinstance(recommendations, dict):
            latest_actions = recommendations.get("latest", [])
            if isinstance(latest_actions, list) and latest_actions:
                lines.append("Latest recommendations:")
                for index, action in enumerate(latest_actions[:3], start=1):
                    lines.append(f"{index}. {action}")
        return "\n".join(lines)

    if result.tool_name == "generate_project_report":
        next_steps = result.data.get("recommended_next_steps", [])
        lines = [
            (
                f"Generated report #{result.data.get('report_id')}: "
                f"{result.data.get('title')}."
            )
        ]
        if isinstance(next_steps, list) and next_steps:
            lines.append("Recommended next steps:")
            for index, step in enumerate(next_steps[:3], start=1):
                lines.append(f"{index}. {step}")
        return "\n".join(lines)

    if result.tool_name == "list_reports":
        reports = result.data.get("reports", [])
        if not reports:
            return "No generated reports exist for this project yet."
        lines = ["Generated reports:"]
        if isinstance(reports, list):
            for report in reports[:10]:
                if not isinstance(report, dict):
                    continue
                lines.append(
                    "- #{id}: {title} ({report_type})".format(
                        id=report.get("report_id"),
                        title=report.get("title"),
                        report_type=report.get("report_type"),
                    )
                )
        return "\n".join(lines)

    if result.tool_name == "explain_latest_report":
        if not result.data.get("report_available", False):
            return str(result.data.get("message") or "No generated reports exist yet.")
        lines = [
            f"Latest report #{result.data.get('report_id')}: {result.data.get('title')}.",
            str(result.data.get("summary") or ""),
        ]
        next_steps = result.data.get("recommended_next_steps", [])
        if isinstance(next_steps, list) and next_steps:
            lines.append("Recommended next steps:")
            for index, step in enumerate(next_steps[:3], start=1):
                lines.append(f"{index}. {step}")
        return "\n".join(line for line in lines if line)

    if result.tool_name == "review_latest_report":
        if not result.data.get("report_available", False):
            return str(result.data.get("message") or "No generated reports exist yet.")
        lines = [f"Report review for #{result.data.get('report_id')}: {result.data.get('title')}."]
        strengths = result.data.get("strengths", [])
        missing_sections = result.data.get("missing_sections", [])
        suggested_edits = result.data.get("suggested_edits", [])
        limitations = result.data.get("limitations", [])
        if isinstance(strengths, list) and strengths:
            lines.append("Strengths: " + "; ".join(str(item) for item in strengths[:3]) + ".")
        if isinstance(missing_sections, list):
            missing_text = (
                "; ".join(str(item) for item in missing_sections[:5])
                if missing_sections
                else "none"
            )
            lines.append(f"Missing sections: {missing_text}.")
        if isinstance(suggested_edits, list) and suggested_edits:
            lines.append("Suggested edits:")
            for index, edit in enumerate(suggested_edits[:4], start=1):
                lines.append(f"{index}. {edit}")
        if isinstance(limitations, list) and limitations:
            lines.append("Review limitations: " + "; ".join(str(item) for item in limitations[:3]) + ".")
        return "\n".join(lines)

    return json.dumps(result.data, indent=2, sort_keys=True, default=str)


def _synthesize_plan_response(
    message: str,
    plan: list[PlanStep],
    results: list[ToolResult],
) -> str:
    completed = sum(1 for step in plan if step.status == "completed")
    failed = len(plan) - completed
    lines = [
        f"I ran a {len(plan)} step project review plan for: {message.strip()}",
        f"Completed {completed} step(s); {failed} step(s) could not complete.",
        "",
        "What I checked:",
    ]
    for step in plan:
        status = "ok" if step.status == "completed" else "needs attention"
        summary = step.result_summary or "No summary returned."
        lines.append(f"{step.step_id}. {step.tool_name} ({status}): {summary}")

    recommendations = _collect_plan_recommendations(results)
    if recommendations:
        lines.extend(["", "Recommended next moves:"])
        for index, recommendation in enumerate(recommendations[:3], start=1):
            lines.append(f"{index}. {recommendation}")
    else:
        lines.extend(
            [
                "",
                "Recommended next move: fill the missing project context above, then ask me to review status again.",
            ]
        )

    lines.append(
        "I did not train models, run simulations, or launch optimization during this review."
    )
    return "\n".join(lines)


def _collect_plan_recommendations(results: list[ToolResult]) -> list[str]:
    recommendations: list[str] = []
    for result in results:
        if not result.success:
            continue
        if result.tool_name == "recommend_next_experiment":
            for recommendation in result.data.get("recommendations", []):
                if not isinstance(recommendation, dict):
                    continue
                inputs = recommendation.get("inputs", {})
                if isinstance(inputs, dict):
                    recommendations.append(
                        "Review simulated candidate {rank}: {temperature} C for {batch_time} min, then validate with domain constraints.".format(
                            rank=recommendation.get("rank", "?"),
                            temperature=inputs.get("temperature_c", "?"),
                            batch_time=inputs.get("batch_time_min", "?"),
                        )
                    )
        if result.tool_name == "explain_latest_model":
            recommendations.extend(
                str(step)
                for step in result.data.get("suggested_next_steps", [])
                if step
            )
        if result.tool_name == "run_project_analysis_workflow":
            workflow_result = result.data.get("result", {})
            if isinstance(workflow_result, dict):
                recommendations.extend(
                    str(step)
                    for step in workflow_result.get("recommended_next_actions", [])
                    if step
                )
    return _unique_strings(recommendations)


def _plan_step_payload(step: PlanStep) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "tool_name": step.tool_name,
        "arguments_json": step.arguments_json,
        "purpose": step.purpose,
        "status": step.status,
        "result_summary": step.result_summary,
    }


def _bounded_text(value: str, limit: int = 260) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _chat_response(
    reply: str,
    *,
    tool_used: bool = False,
    tool_name: str | None = None,
    tool_result: dict[str, Any] | None = None,
    plan_executed: bool = False,
    steps_summary: list[dict[str, Any]] | None = None,
    tools_used: list[str] | None = None,
    pending_action: dict[str, Any] | None = None,
) -> ChatResponse:
    return ChatResponse(
        reply=reply,
        message=reply,
        tool_used=tool_used,
        tool_name=tool_name,
        tool_result=tool_result,
        plan_executed=plan_executed,
        steps_summary=steps_summary,
        tools_used=tools_used,
        pending_action=pending_action,
    )


def _structured_tool_result(result: ToolResult) -> dict[str, Any]:
    if not result.success:
        return {
            "tool_name": result.tool_name,
            "success": False,
            "error": result.error,
        }

    if result.tool_name == "get_dataset_summary":
        dataset = result.data.get("dataset", {})
        column_names = _string_list(result.data.get("column_names", []))
        column_types = result.data.get("column_types", {})
        return {
            "tool_name": result.tool_name,
            "success": True,
            "dataset_id": dataset.get("id"),
            "filename": dataset.get("filename"),
            "rows": dataset.get("row_count"),
            "column_count": dataset.get("column_count"),
            "columns": column_names,
            "numeric_columns": _columns_by_type(column_types, numeric=True),
            "categorical_columns": _columns_by_type(column_types, numeric=False),
            "missing_values": result.data.get("missing_values", {}),
            "preview": result.data.get("preview", []),
            "memory_notes": result.data.get("memory_notes", []),
        }

    if result.tool_name == "show_missing_values":
        dataset = result.data.get("dataset", {})
        return {
            "tool_name": result.tool_name,
            "success": True,
            "dataset_id": dataset.get("id"),
            "filename": dataset.get("filename"),
            "rows": dataset.get("row_count"),
            "column_count": dataset.get("column_count"),
            "missing_values": result.data.get("missing_values", {}),
            "columns_with_missing": result.data.get("columns_with_missing", {}),
            "total_missing_values": result.data.get("total_missing_values", 0),
            "memory_notes": result.data.get("memory_notes", []),
        }

    if result.tool_name == "train_baseline_model":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "model_run_id": result.data.get("model_run_id"),
            "dataset_id": result.data.get("dataset_id"),
            "target_column": result.data.get("target_column"),
            "task_type": result.data.get("task_type"),
            "metrics": result.data.get("metrics", {}),
            "top_features": result.data.get("feature_importance", []),
            "memory_notes": result.data.get("memory_notes", []),
        }

    if result.tool_name == "explain_latest_model":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "model_available": result.data.get("model_available", False),
            "message": result.data.get("message"),
            "model_run_id": result.data.get("model_run_id"),
            "dataset_id": result.data.get("dataset_id"),
            "dataset": result.data.get("dataset"),
            "target_column": result.data.get("target_column"),
            "task_type": result.data.get("task_type"),
            "model_type": result.data.get("model_type"),
            "metrics": result.data.get("metrics", {}),
            "top_features": result.data.get("top_features", []),
            "limitations": result.data.get("limitations", []),
            "suggested_next_steps": result.data.get("suggested_next_steps", []),
        }

    if result.tool_name == "answer_document_question":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "answer": result.data.get("answer"),
            "sources": result.data.get("sources", []),
        }

    if result.tool_name == "run_batch_reactor_simulation":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "simulation_run_id": result.data.get("simulation_run_id"),
            "simulation_type": result.data.get("simulation_type"),
            "input": result.data.get("input", {}),
            "time_grid": result.data.get("time_grid", []),
            "CA_profile": result.data.get("CA_profile", []),
            "CB_profile": result.data.get("CB_profile", []),
            "CC_profile": result.data.get("CC_profile", []),
            "final_yield": result.data.get("final_yield"),
            "final_impurity": result.data.get("final_impurity"),
            "conversion": result.data.get("conversion"),
            "rate_constants": result.data.get("rate_constants", {}),
            "note": result.data.get("note"),
        }

    if result.tool_name == "list_simulation_runs":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "simulation_runs": result.data.get("simulation_runs", []),
            "count": result.data.get("count", 0),
        }

    if result.tool_name == "explain_latest_simulation":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "simulation_available": result.data.get("simulation_available", False),
            "message": result.data.get("message"),
            "simulation_run_id": result.data.get("simulation_run_id"),
            "simulation_type": result.data.get("simulation_type"),
            "input": result.data.get("input", {}),
            "result": result.data.get("result", {}),
            "interpretation": result.data.get("interpretation"),
            "model_note": result.data.get("model_note"),
        }

    if result.tool_name == "compare_simulation_runs":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "comparison_available": result.data.get("comparison_available", False),
            "message": result.data.get("message"),
            "baseline": result.data.get("baseline"),
            "candidate": result.data.get("candidate"),
            "input_differences": result.data.get("input_differences", {}),
            "result_differences": result.data.get("result_differences", {}),
            "interpretation": result.data.get("interpretation"),
            "model_note": result.data.get("model_note"),
        }

    if result.tool_name == "optimize_batch_reactor":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "optimization_run_id": result.data.get("optimization_run_id"),
            "optimization_type": result.data.get("optimization_type"),
            "objective": result.data.get("objective"),
            "constraints": result.data.get("constraints", {}),
            "search_space": result.data.get("search_space", {}),
            "best_inputs": result.data.get("best_inputs", {}),
            "best_final_yield": result.data.get("best_final_yield"),
            "best_final_impurity": result.data.get("best_final_impurity"),
            "best_conversion": result.data.get("best_conversion"),
            "objective_value": result.data.get("objective_value"),
            "top_candidates": result.data.get("top_candidates", []),
            "evaluated_candidates": result.data.get("evaluated_candidates"),
            "feasible_candidates": result.data.get("feasible_candidates"),
            "note": result.data.get("note"),
        }

    if result.tool_name == "explain_latest_optimization":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "optimization_available": result.data.get("optimization_available", False),
            "message": result.data.get("message"),
            "optimization_run_id": result.data.get("optimization_run_id"),
            "optimization_type": result.data.get("optimization_type"),
            "objective": result.data.get("objective"),
            "constraints": result.data.get("constraints", {}),
            "search_space": result.data.get("search_space", {}),
            "best_inputs": result.data.get("best_inputs", {}),
            "best_final_yield": result.data.get("best_final_yield"),
            "best_final_impurity": result.data.get("best_final_impurity"),
            "best_conversion": result.data.get("best_conversion"),
            "objective_value": result.data.get("objective_value"),
            "top_candidates": result.data.get("top_candidates", []),
            "evaluated_candidates": result.data.get("evaluated_candidates"),
            "feasible_candidates": result.data.get("feasible_candidates"),
            "explanation": result.data.get("explanation"),
            "simulation_comparison": result.data.get("simulation_comparison"),
            "note": result.data.get("model_note"),
        }

    if result.tool_name == "list_optimization_runs":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "optimization_runs": result.data.get("optimization_runs", []),
            "count": result.data.get("count", 0),
        }

    if result.tool_name == "recommend_next_experiment":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "recommendation_available": result.data.get(
                "recommendation_available",
                False,
            ),
            "message": result.data.get("message"),
            "optimization_run_id": result.data.get("optimization_run_id"),
            "optimization_type": result.data.get("optimization_type"),
            "recommendations": result.data.get("recommendations", []),
            "recommendation_count": result.data.get("recommendation_count", 0),
            "note": result.data.get("note"),
        }

    if result.tool_name == "run_project_analysis_workflow":
        workflow_result = result.data.get("result", {})
        return {
            "tool_name": result.tool_name,
            "success": True,
            "workflow_run_id": result.data.get("workflow_run_id"),
            "workflow_type": result.data.get("workflow_type"),
            "status": result.data.get("status"),
            "steps": result.data.get("steps", []),
            "project_name": workflow_result.get("project_name")
            if isinstance(workflow_result, dict)
            else None,
            "summary": workflow_result.get("summary")
            if isinstance(workflow_result, dict)
            else None,
            "current_assets": workflow_result.get("current_assets", {})
            if isinstance(workflow_result, dict)
            else {},
            "gaps": workflow_result.get("gaps", [])
            if isinstance(workflow_result, dict)
            else [],
            "recommended_next_actions": workflow_result.get(
                "recommended_next_actions",
                [],
            )
            if isinstance(workflow_result, dict)
            else [],
        }

    if result.tool_name == "list_workflow_runs":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "workflow_runs": result.data.get("workflow_runs", []),
            "count": result.data.get("count", 0),
        }

    if result.tool_name == "explain_latest_workflow":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "workflow_available": result.data.get("workflow_available", False),
            "message": result.data.get("message"),
            "workflow_run_id": result.data.get("workflow_run_id"),
            "workflow_type": result.data.get("workflow_type"),
            "status": result.data.get("status"),
            "created_at": result.data.get("created_at"),
            "completed_at": result.data.get("completed_at"),
            "steps": result.data.get("steps", []),
            "summary": result.data.get("summary"),
            "current_assets": result.data.get("current_assets", {}),
            "gaps": result.data.get("gaps", []),
            "recommended_next_actions": result.data.get(
                "recommended_next_actions",
                [],
            ),
        }

    if result.tool_name == "compare_workflow_runs":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "comparison_available": result.data.get("comparison_available", False),
            "message": result.data.get("message"),
            "latest_workflow": result.data.get("latest_workflow"),
            "previous_workflow": result.data.get("previous_workflow"),
            "status_comparison": result.data.get("status_comparison", {}),
            "major_project_changes": result.data.get("major_project_changes", []),
            "gaps": result.data.get("gaps", {}),
            "workflow_recommendations": result.data.get("recommendations", {}),
        }

    if result.tool_name == "generate_project_report":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "report_id": result.data.get("report_id"),
            "project_id": result.data.get("project_id"),
            "report_type": result.data.get("report_type"),
            "title": result.data.get("title"),
            "content_markdown": result.data.get("content_markdown"),
            "source_summary": result.data.get("source_summary", {}),
            "recommended_next_steps": result.data.get(
                "recommended_next_steps",
                [],
            ),
            "created_at": result.data.get("created_at"),
        }

    if result.tool_name == "list_reports":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "reports": result.data.get("reports", []),
            "count": result.data.get("count", 0),
        }

    if result.tool_name == "explain_latest_report":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "report_available": result.data.get("report_available", False),
            "message": result.data.get("message"),
            "report_id": result.data.get("report_id"),
            "title": result.data.get("title"),
            "report_type": result.data.get("report_type"),
            "created_at": result.data.get("created_at"),
            "sections": result.data.get("sections", []),
            "summary": result.data.get("summary"),
            "recommended_next_steps": result.data.get(
                "recommended_next_steps",
                [],
            ),
        }

    if result.tool_name == "review_latest_report":
        return {
            "tool_name": result.tool_name,
            "success": True,
            "report_available": result.data.get("report_available", False),
            "message": result.data.get("message"),
            "report_id": result.data.get("report_id"),
            "title": result.data.get("title"),
            "report_type": result.data.get("report_type"),
            "created_at": result.data.get("created_at"),
            "sections": result.data.get("sections", []),
            "strengths": result.data.get("strengths", []),
            "missing_sections": result.data.get("missing_sections", []),
            "weak_sections": result.data.get("weak_sections", []),
            "suggested_edits": result.data.get("suggested_edits", []),
            "limitations": result.data.get("limitations", []),
        }

    return {
        "tool_name": result.tool_name,
        "success": True,
        **result.data,
    }


def _pending_action_payload(
    tool_name: str,
    arguments: dict[str, Any],
    missing_fields: list[str],
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "missing_fields": missing_fields,
    }


def _memory_notes_text(result: ToolResult) -> str:
    notes = result.data.get("memory_notes", [])
    if not isinstance(notes, list) or not notes:
        return ""
    return " ".join(str(note) for note in notes if note) + " "


def _format_asset_status(assets: Any) -> str:
    if not isinstance(assets, dict) or not assets:
        return "no structured asset status returned."
    labels = {
        "dataset_available": "dataset",
        "document_context_available": "documents",
        "model_available": "model",
        "simulation_available": "simulation",
        "optimization_available": "optimization",
    }
    parts = [
        f"{label}: {'yes' if assets.get(key) else 'no'}"
        for key, label in labels.items()
    ]
    return ", ".join(parts) + "."


def _format_asset_change(change: dict[str, Any]) -> str:
    asset = str(change.get("asset") or "asset").replace("_available", "")
    asset = asset.replace("_", " ")
    change_type = change.get("change")
    if change_type == "became_available":
        return f"{asset} became available"
    if change_type == "became_missing":
        return f"{asset} became missing"
    return f"{asset} changed"


def _format_project_memory(memories: Any) -> str:
    memory_items = memories if isinstance(memories, list) else []
    memory_by_key = {
        str(memory.get("key")): memory.get("value")
        for memory in memory_items
        if isinstance(memory, dict) and memory.get("key")
    }

    lines = ["I currently remember:"]

    latest_document = memory_by_key.get("latest_document_filename")
    document_count = memory_by_key.get("document_count")
    if latest_document:
        lines.append(f"- Latest document: {latest_document}")
    else:
        lines.append("- No documents uploaded yet")
    if document_count is not None:
        lines.append(f"- Documents uploaded: {document_count}")

    latest_dataset = memory_by_key.get("latest_dataset_filename")
    dataset_count = memory_by_key.get("dataset_count")
    if latest_dataset:
        lines.append(f"- Latest dataset: {latest_dataset}")
    else:
        lines.append("- No datasets uploaded yet")
    if dataset_count is not None:
        lines.append(f"- Datasets uploaded: {dataset_count}")

    latest_model_run_id = memory_by_key.get("latest_model_run_id")
    if latest_model_run_id:
        task_type = memory_by_key.get("latest_task_type")
        model_line = f"- Latest model run: #{latest_model_run_id}"
        if task_type:
            model_line += f" ({task_type})"
        lines.append(model_line)
    else:
        lines.append("- No trained models yet")

    selected_target_column = memory_by_key.get("selected_target_column")
    if selected_target_column:
        lines.append(f"- Selected target column: {selected_target_column}")

    latest_optimization_run_id = memory_by_key.get("latest_optimization_run_id")
    if latest_optimization_run_id:
        optimization_type = memory_by_key.get("latest_optimization_type")
        optimization_line = f"- Latest optimization run: #{latest_optimization_run_id}"
        if optimization_type:
            optimization_line += f" ({optimization_type})"
        lines.append(optimization_line)

    recommended_experiment_count = memory_by_key.get("recommended_experiment_count")
    if recommended_experiment_count is not None:
        lines.append(
            f"- Recommended simulated experiments: {recommended_experiment_count}"
        )

    displayed_keys = {
        "latest_document_id",
        "latest_document_filename",
        "document_count",
        "latest_dataset_id",
        "latest_dataset_filename",
        "dataset_count",
        "latest_model_run_id",
        "latest_task_type",
        "target_column",
        "selected_target_column",
        "latest_optimization_run_id",
        "latest_optimization_type",
        "latest_recommended_experiment",
        "recommended_experiment_count",
        "project_summary",
    }
    other_memories = [
        memory
        for memory in memory_items
        if isinstance(memory, dict) and memory.get("key") not in displayed_keys
    ]
    for memory in other_memories:
        lines.append(f"- {memory.get('key')}: {memory.get('value')}")

    return "\n".join(lines)


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _columns_by_type(column_types: Any, *, numeric: bool) -> list[str]:
    if not isinstance(column_types, dict):
        return []

    numeric_tokens = ("int", "float", "double", "decimal", "number")
    columns: list[str] = []
    for column, dtype in column_types.items():
        dtype_text = str(dtype).casefold()
        is_numeric = any(token in dtype_text for token in numeric_tokens)
        if is_numeric == numeric:
            columns.append(str(column))
    return columns


def _available_columns_for_training(
    db: Session,
    project_id: int,
    arguments: dict[str, Any],
    memory_values: dict[str, Any] | None = None,
) -> list[str]:
    try:
        dataset = resolve_dataset(
            db,
            project_id,
            arguments.get("dataset_id"),
            bool(arguments.get("latest", True)),
            memory_values,
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
