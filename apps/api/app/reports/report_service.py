import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.db.models import Document
from app.db.models import ModelRun
from app.db.models import OptimizationRun
from app.db.models import Project
from app.db.models import Report
from app.db.models import SimulationRun
from app.optimization.optimization_service import optimization_run_payload
from app.reports.schemas import ReportListItem, ReportRead
from app.services.documents import list_project_documents
from app.services.memory_service import (
    get_memory,
    list_memory,
    memory_records_to_dict,
    update_project_summary,
    upsert_memory,
)
from app.services.model_training import list_project_model_runs
from app.services.projects import get_project
from app.simulation.simulation_service import simulation_run_payload
from app.workflows.workflow_service import list_project_workflow_runs, workflow_run_payload


PROJECT_REPORT_TYPE = "project_report"


def generate_project_report(db: Session, project_id: int) -> Report:
    project = get_project(db, project_id)
    if project is None:
        raise ValueError("Project not found.")

    source_summary = build_project_report_source_summary(db, project)
    title = _report_title(project)
    content_markdown = _build_report_markdown(title, source_summary)
    report = Report(
        project_id=project_id,
        report_type=PROJECT_REPORT_TYPE,
        title=title,
        content_markdown=content_markdown,
        source_summary_json=json.dumps(source_summary, sort_keys=True, default=str),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    upsert_memory(
        db,
        project_id,
        "latest_report_id",
        report.id,
        memory_type="report",
        source="project_report_generation",
    )
    upsert_memory(
        db,
        project_id,
        "latest_report_title",
        report.title,
        memory_type="report",
        source="project_report_generation",
    )
    update_project_summary(db, project_id)
    return report


def list_project_reports(db: Session, project_id: int) -> list[Report]:
    result = db.execute(
        select(Report)
        .where(Report.project_id == project_id)
        .order_by(Report.created_at.desc(), Report.id.desc())
    )
    return list(result.scalars().all())


def get_project_report(db: Session, project_id: int, report_id: int) -> Report | None:
    report = db.get(Report, report_id)
    if report is None or report.project_id != project_id:
        return None
    return report


def latest_project_report(
    db: Session,
    project_id: int,
    report_id: int | None = None,
) -> Report | None:
    if report_id is not None:
        report = get_project_report(db, project_id, report_id)
        if report is not None:
            return report

    remembered_report_id = _remembered_latest_report_id(db, project_id)
    if remembered_report_id is not None:
        report = get_project_report(db, project_id, remembered_report_id)
        if report is not None:
            return report

    reports = list_project_reports(db, project_id)
    return reports[0] if reports else None


def report_to_read_data(report: Report) -> ReportRead:
    return ReportRead(
        id=report.id,
        project_id=report.project_id,
        report_type=report.report_type,
        title=report.title,
        content_markdown=report.content_markdown,
        source_summary=_loads_dict(report.source_summary_json),
        created_at=report.created_at,
    )


def report_to_list_item(report: Report) -> ReportListItem:
    return ReportListItem(
        id=report.id,
        project_id=report.project_id,
        report_type=report.report_type,
        title=report.title,
        created_at=report.created_at,
    )


def report_tool_payload(report: Report) -> dict[str, Any]:
    source_summary = _loads_dict(report.source_summary_json)
    next_steps = source_summary.get("recommended_next_steps", [])
    return {
        "report_id": report.id,
        "project_id": report.project_id,
        "report_type": report.report_type,
        "title": report.title,
        "content_markdown": report.content_markdown,
        "source_summary": source_summary,
        "recommended_next_steps": next_steps if isinstance(next_steps, list) else [],
        "created_at": report.created_at.isoformat(),
    }


def report_summary_payload(report: Report) -> dict[str, Any]:
    source_summary = _loads_dict(report.source_summary_json)
    return {
        "report_id": report.id,
        "project_id": report.project_id,
        "report_type": report.report_type,
        "title": report.title,
        "created_at": report.created_at.isoformat(),
        "section_count": len(_markdown_sections(report.content_markdown)),
        "source_summary": {
            "project": source_summary.get("project"),
            "recommended_next_steps": source_summary.get("recommended_next_steps", []),
        },
    }


def build_project_report_source_summary(
    db: Session,
    project: Project,
) -> dict[str, Any]:
    update_project_summary(db, project.id)
    memory = memory_records_to_dict(list_memory(db, project.id))
    datasets = _list_latest_rows(db, Dataset, project.id)
    documents = list_project_documents(db, project.id)
    model_runs = list_project_model_runs(db, project.id)
    simulation_runs = _list_latest_rows(db, SimulationRun, project.id)
    optimization_runs = _list_latest_rows(db, OptimizationRun, project.id)
    workflow_runs = list_project_workflow_runs(db, project.id, limit=1)

    latest_workflow = (
        workflow_run_payload(workflow_runs[0]) if workflow_runs else None
    )
    workflow_result = (
        latest_workflow.get("result", {})
        if isinstance(latest_workflow, dict)
        else {}
    )
    workflow_recommendations = (
        workflow_result.get("recommended_next_actions", [])
        if isinstance(workflow_result, dict)
        else []
    )
    recommended_next_steps = (
        workflow_recommendations
        if isinstance(workflow_recommendations, list) and workflow_recommendations
        else _fallback_next_steps(datasets, model_runs, simulation_runs, optimization_runs)
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
        },
        "memory": {
            "project_summary": memory.get("project_summary"),
            "selected_target_column": memory.get("selected_target_column"),
            "latest_dataset_filename": memory.get("latest_dataset_filename"),
            "latest_document_filename": memory.get("latest_document_filename"),
            "latest_model_run_id": memory.get("latest_model_run_id"),
            "latest_workflow_run_id": memory.get("latest_workflow_run_id"),
        },
        "datasets": [_dataset_summary(dataset) for dataset in datasets[:10]],
        "documents": [_document_summary(document) for document in documents[:10]],
        "latest_model_run": (
            _model_run_summary(db, model_runs[0]) if model_runs else None
        ),
        "latest_simulation_run": (
            _simulation_run_summary(simulation_runs[0]) if simulation_runs else None
        ),
        "latest_optimization_run": (
            _optimization_run_summary(optimization_runs[0])
            if optimization_runs
            else None
        ),
        "latest_workflow_run": latest_workflow,
        "recommended_next_steps": [str(step) for step in recommended_next_steps[:3]],
    }


def _build_report_markdown(title: str, source: dict[str, Any]) -> str:
    project = _dict_value(source.get("project"))
    memory = _dict_value(source.get("memory"))
    datasets = _list_value(source.get("datasets"))
    documents = _list_value(source.get("documents"))
    model_run = _dict_value(source.get("latest_model_run"))
    simulation_run = _dict_value(source.get("latest_simulation_run"))
    optimization_run = _dict_value(source.get("latest_optimization_run"))
    workflow_run = _dict_value(source.get("latest_workflow_run"))
    workflow_result = _dict_value(workflow_run.get("result"))
    next_steps = [str(step) for step in _list_value(source.get("recommended_next_steps"))]

    sections = [
        f"# {title}",
        "## Project Overview",
        _project_overview(project, memory),
        "## Available Data",
        _datasets_markdown(datasets),
        "## Uploaded Documents",
        _documents_markdown(documents),
        "## Model Results",
        _model_markdown(model_run),
        "## Simulation Results",
        _simulation_markdown(simulation_run),
        "## Optimization Results",
        _optimization_markdown(optimization_run),
        "## Workflow Recommendations",
        _workflow_markdown(workflow_run, workflow_result),
        "## Limitations",
        _limitations_markdown(datasets, documents, model_run, simulation_run, optimization_run),
        "## Recommended Next Steps",
        _bullet_list(next_steps) if next_steps else "No next steps are available yet.",
    ]
    return "\n\n".join(sections).strip() + "\n"


def _report_title(project: Project) -> str:
    return f"{project.name} Technical Summary"


def _project_overview(project: dict[str, Any], memory: dict[str, Any]) -> str:
    lines = [
        f"- Project: {project.get('name') or 'Untitled project'}",
    ]
    description = project.get("description")
    if description:
        lines.append(f"- Description: {description}")
    project_summary = memory.get("project_summary")
    if project_summary:
        lines.append(f"- Current memory summary: {project_summary}")
    return "\n".join(lines)


def _datasets_markdown(datasets: list[Any]) -> str:
    if not datasets:
        return "No datasets are saved in this project yet."
    lines = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        columns = dataset.get("column_names", [])
        column_text = ", ".join(str(column) for column in columns[:12])
        lines.append(
            f"- {dataset.get('filename')} (id {dataset.get('id')}): "
            f"{dataset.get('row_count')} rows, {dataset.get('column_count')} columns."
        )
        if column_text:
            lines.append(f"  Columns: {column_text}.")
        missing = dataset.get("missing_values", {})
        if isinstance(missing, dict) and missing:
            missing_text = ", ".join(
                f"{column}: {count}" for column, count in list(missing.items())[:8]
            )
            lines.append(f"  Missing values: {missing_text}.")
    return "\n".join(lines)


def _documents_markdown(documents: list[Any]) -> str:
    if not documents:
        return "No documents are saved in this project yet."
    return "\n".join(
        "- {filename} (id {id}): {mime_type}, {file_size} bytes, extracted text: {has_extracted_text}.".format(
            **document
        )
        for document in documents
        if isinstance(document, dict)
    )


def _model_markdown(model_run: dict[str, Any]) -> str:
    if not model_run:
        return "No model runs are saved yet."
    lines = [
        f"- Latest model run: #{model_run.get('id')}",
        f"- Model: {model_run.get('model_type')} ({model_run.get('task_type')})",
        f"- Target column: {model_run.get('target_column')}",
    ]
    metrics = _dict_value(model_run.get("metrics"))
    if metrics:
        lines.append(
            "- Metrics: "
            + ", ".join(f"{key}: {value}" for key, value in metrics.items())
        )
    top_features = _list_value(model_run.get("top_features"))
    feature_names = [
        str(feature.get("feature"))
        for feature in top_features[:5]
        if isinstance(feature, dict) and feature.get("feature")
    ]
    if feature_names:
        lines.append("- Top features: " + ", ".join(feature_names))
    return "\n".join(lines)


def _simulation_markdown(simulation_run: dict[str, Any]) -> str:
    if not simulation_run:
        return "No simulation runs are saved yet."
    result = _dict_value(simulation_run.get("result"))
    return "\n".join(
        [
            f"- Latest simulation run: #{simulation_run.get('id')}",
            f"- Type: {simulation_run.get('simulation_type')}",
            f"- Final yield: {result.get('final_yield')}",
            f"- Final impurity: {result.get('final_impurity')}",
            f"- Conversion: {result.get('conversion')}",
        ]
    )


def _optimization_markdown(optimization_run: dict[str, Any]) -> str:
    if not optimization_run:
        return "No optimization runs are saved yet."
    return "\n".join(
        [
            f"- Latest optimization run: #{optimization_run.get('id')}",
            f"- Type: {optimization_run.get('optimization_type')}",
            f"- Objective: {optimization_run.get('objective')}",
            f"- Best final yield: {optimization_run.get('best_final_yield')}",
            f"- Best final impurity: {optimization_run.get('best_final_impurity')}",
            f"- Objective value: {optimization_run.get('objective_value')}",
        ]
    )


def _workflow_markdown(
    workflow_run: dict[str, Any],
    workflow_result: dict[str, Any],
) -> str:
    if not workflow_run:
        return "No workflow runs are saved yet."
    lines = [
        f"- Latest workflow run: #{workflow_run.get('workflow_run_id')}",
        f"- Status: {workflow_run.get('status')}",
    ]
    summary = workflow_result.get("summary")
    if summary:
        lines.append(f"- Summary: {summary}")
    recommendations = _list_value(workflow_result.get("recommended_next_actions"))
    if recommendations:
        lines.append("- Recommendations:")
        lines.extend(f"  - {recommendation}" for recommendation in recommendations[:3])
    return "\n".join(lines)


def _limitations_markdown(
    datasets: list[Any],
    documents: list[Any],
    model_run: dict[str, Any],
    simulation_run: dict[str, Any],
    optimization_run: dict[str, Any],
) -> str:
    limitations = [
        "This report is generated deterministically from saved workspace state.",
        "It does not include external validation or unseen project artifacts.",
    ]
    if not datasets:
        limitations.append("No saved dataset is available for quantitative review.")
    if not documents:
        limitations.append("No uploaded documents are available for context.")
    if model_run:
        limitations.append("Model feature importance is predictive association, not causation.")
    if simulation_run or optimization_run:
        limitations.append("Simulation and optimization outputs use the simplified benchmark model, not calibrated real-world chemistry.")
    return _bullet_list(limitations)


def _fallback_next_steps(
    datasets: list[Dataset],
    model_runs: list[ModelRun],
    simulation_runs: list[SimulationRun],
    optimization_runs: list[OptimizationRun],
) -> list[str]:
    if not datasets:
        return [
            "Upload a CSV dataset for the project.",
            "Upload supporting papers, SOPs, or notes.",
            "Run project analysis again after adding data.",
        ]
    if not model_runs:
        return [
            "Select a target column and train a baseline model.",
            "Review missing values and column definitions.",
            "Ask document-grounded questions about the dataset variables.",
        ]
    if not simulation_runs:
        return [
            "Review model metrics and top features.",
            "Run a simulation for a representative scenario.",
            "Compare model findings against uploaded document context.",
        ]
    if not optimization_runs:
        return [
            "Run a transparent optimization over the simulation benchmark.",
            "Compare optimized settings with the latest simulation.",
            "Review recommendations with domain constraints.",
        ]
    return [
        "Compare model, simulation, optimization, and document evidence.",
        "Choose the next candidate experiment for expert review.",
        "Record new results and regenerate this report.",
    ]


def _dataset_summary(dataset: Dataset) -> dict[str, Any]:
    profile = _dataset_profile(dataset)
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "created_at": dataset.created_at.isoformat(),
        **profile,
    }


def _dataset_profile(dataset: Dataset) -> dict[str, Any]:
    try:
        rows = json.loads(dataset.raw_data_json)
    except json.JSONDecodeError:
        return {"column_names": [], "missing_values": {}}
    if not isinstance(rows, list):
        return {"column_names": [], "missing_values": {}}
    dataframe = pd.DataFrame(rows)
    return {
        "column_names": [str(column) for column in dataframe.columns[:20]],
        "missing_values": {
            str(column): int(value)
            for column, value in dataframe.isna().sum().items()
            if int(value) > 0
        },
    }


def _document_summary(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "filename": document.filename,
        "mime_type": document.mime_type,
        "file_size": document.file_size,
        "has_extracted_text": document.extracted_text_path is not None,
        "created_at": document.created_at.isoformat(),
    }


def _model_run_summary(db: Session, model_run: ModelRun) -> dict[str, Any]:
    dataset = db.get(Dataset, model_run.dataset_id) if model_run.dataset_id else None
    return {
        "id": model_run.id,
        "dataset": _dataset_reference(dataset),
        "target_column": model_run.target_column,
        "task_type": model_run.task_type,
        "model_type": model_run.model_type,
        "metrics": _loads_dict(model_run.metrics_json),
        "top_features": _loads_list(model_run.feature_importance_json)[:5],
        "created_at": model_run.created_at.isoformat(),
    }


def _simulation_run_summary(simulation_run: SimulationRun) -> dict[str, Any]:
    payload = simulation_run_payload(simulation_run)
    return {
        "id": simulation_run.id,
        "simulation_type": simulation_run.simulation_type,
        "input": payload.get("input", {}),
        "result": payload.get("result", {}),
        "created_at": simulation_run.created_at.isoformat(),
    }


def _optimization_run_summary(optimization_run: OptimizationRun) -> dict[str, Any]:
    payload = optimization_run_payload(optimization_run)
    return {
        "id": optimization_run.id,
        "optimization_type": optimization_run.optimization_type,
        "objective": optimization_run.objective,
        "constraints": payload.get("constraints", {}),
        "best_inputs": payload.get("best_inputs", {}),
        "best_final_yield": payload.get("best_final_yield"),
        "best_final_impurity": payload.get("best_final_impurity"),
        "objective_value": payload.get("objective_value"),
        "created_at": optimization_run.created_at.isoformat(),
    }


def _list_latest_rows(db: Session, model: type[Any], project_id: int) -> list[Any]:
    result = db.execute(
        select(model)
        .where(model.project_id == project_id)
        .order_by(model.created_at.desc(), model.id.desc())
    )
    return list(result.scalars().all())


def _dataset_reference(dataset: Dataset | None) -> dict[str, Any] | None:
    if dataset is None:
        return None
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
    }


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _loads_dict(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _loads_list(value: str) -> list[Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return decoded if isinstance(decoded, list) else []


def _remembered_latest_report_id(db: Session, project_id: int) -> int | None:
    memory = get_memory(db, project_id, "latest_report_id")
    if memory is None:
        return None
    value = _loads_value(memory.value_json)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _loads_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _markdown_sections(markdown: str) -> list[str]:
    sections: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            sections.append(line[3:].strip())
    return sections


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
