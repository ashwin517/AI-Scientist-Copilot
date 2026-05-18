import json
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.db.models import Document
from app.db.models import ModelRun
from app.db.models import OptimizationRun
from app.db.models import Project
from app.db.models import SimulationRun
from app.optimization.optimization_service import optimization_run_payload
from app.services.documents import list_project_documents
from app.services.memory_service import (
    list_memory,
    memory_records_to_dict,
    update_project_summary,
    upsert_memory,
)
from app.services.model_training import list_project_model_runs
from app.services.projects import get_project
from app.simulation.simulation_service import simulation_run_payload
from app.workflows.workflow_service import complete_workflow_run, create_workflow_run


PROJECT_ANALYSIS_WORKFLOW_TYPE = "project_analysis"


def run_project_analysis_workflow(
    db: Session,
    project_id: int,
) -> dict[str, Any]:
    project = get_project(db, project_id)
    if project is None:
        raise ValueError("Project not found.")

    workflow_run = create_workflow_run(
        db,
        project_id,
        PROJECT_ANALYSIS_WORKFLOW_TYPE,
    )
    steps: list[dict[str, Any]] = []

    try:
        update_project_summary(db, project_id)
        memory = memory_records_to_dict(list_memory(db, project_id))
        memory_step = _memory_step(project, memory)
        steps.append(memory_step)

        datasets = _list_latest_rows(db, Dataset, project_id)
        dataset_step = _dataset_step(datasets[0] if datasets else None, len(datasets))
        steps.append(dataset_step)

        documents = list_project_documents(db, project_id)
        document_step = _document_step(documents[0] if documents else None, len(documents))
        steps.append(document_step)

        model_runs = list_project_model_runs(db, project_id)
        model_step = _model_step(db, model_runs[0] if model_runs else None)
        steps.append(model_step)

        simulation_runs = _list_latest_rows(db, SimulationRun, project_id)
        simulation_step = _simulation_step(simulation_runs[0] if simulation_runs else None)
        steps.append(simulation_step)

        optimization_runs = _list_latest_rows(db, OptimizationRun, project_id)
        optimization_step = _optimization_step(
            optimization_runs[0] if optimization_runs else None
        )
        steps.append(optimization_step)

        result = _build_project_analysis_result(
            project=project,
            memory=memory,
            dataset_step=dataset_step,
            document_step=document_step,
            model_step=model_step,
            simulation_step=simulation_step,
            optimization_step=optimization_step,
        )
        steps.append(
            {
                "name": "generate_recommended_next_actions",
                "status": "completed",
                "summary": "Generated three recommended next actions from current project state.",
                "data": {"recommended_next_actions": result["recommended_next_actions"]},
            }
        )

        completed_run = complete_workflow_run(
            db,
            workflow_run,
            steps=steps,
            result=result,
        )
        result["workflow_run_id"] = completed_run.id
        result["workflow_status"] = completed_run.status

        upsert_memory(
            db,
            project_id,
            "latest_workflow_run_id",
            completed_run.id,
            memory_type="workflow",
            source="project_analysis_workflow",
        )
        upsert_memory(
            db,
            project_id,
            "latest_project_analysis_summary",
            result["summary"],
            memory_type="workflow",
            source="project_analysis_workflow",
        )
        return {
            "workflow_run_id": completed_run.id,
            "workflow_type": completed_run.workflow_type,
            "status": completed_run.status,
            "steps": steps,
            "result": result,
        }
    except Exception:
        complete_workflow_run(
            db,
            workflow_run,
            steps=steps,
            result={"error": "Project analysis workflow failed."},
            status="failed",
        )
        raise


def _memory_step(project: Project, memory: dict[str, Any]) -> dict[str, Any]:
    summary = memory.get("project_summary")
    if not isinstance(summary, str) or not summary:
        summary = f"Project {project.name} has no detailed summary memory yet."
    return {
        "name": "load_project_memory",
        "status": "completed",
        "summary": summary,
        "data": {
            "project_name": project.name,
            "project_description": project.description,
            "memory_keys": sorted(memory),
            "project_summary": summary,
        },
    }


def _dataset_step(dataset: Dataset | None, dataset_count: int) -> dict[str, Any]:
    if dataset is None:
        return _skipped_step("summarize_uploaded_datasets", "No datasets are uploaded yet.")

    summary = f"Latest dataset {dataset.filename} has {dataset.row_count} rows and {dataset.column_count} columns."
    data = {
        "dataset_count": dataset_count,
        "latest_dataset": {
            "id": dataset.id,
            "filename": dataset.filename,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "created_at": dataset.created_at.isoformat(),
        },
    }
    data["latest_dataset"].update(_dataset_profile(dataset))
    return {
        "name": "summarize_uploaded_datasets",
        "status": "completed",
        "summary": summary,
        "data": data,
    }


def _document_step(document: Document | None, document_count: int) -> dict[str, Any]:
    if document is None:
        return _skipped_step("summarize_documents", "No documents are uploaded yet.")

    return {
        "name": "summarize_documents",
        "status": "completed",
        "summary": f"Latest document is {document.filename}; {document_count} document(s) are saved.",
        "data": {
            "document_count": document_count,
            "latest_document": {
                "id": document.id,
                "filename": document.filename,
                "mime_type": document.mime_type,
                "file_size": document.file_size,
                "has_extracted_text": document.extracted_text_path is not None,
                "created_at": document.created_at.isoformat(),
            },
        },
    }


def _model_step(db: Session, model_run: ModelRun | None) -> dict[str, Any]:
    if model_run is None:
        return _skipped_step("summarize_latest_model_run", "No model runs are saved yet.")

    dataset = db.get(Dataset, model_run.dataset_id) if model_run.dataset_id else None
    return {
        "name": "summarize_latest_model_run",
        "status": "completed",
        "summary": (
            f"Latest model run #{model_run.id} is a {model_run.model_type} "
            f"{model_run.task_type} model predicting {model_run.target_column}."
        ),
        "data": {
            "model_run_id": model_run.id,
            "dataset": _dataset_metadata(dataset),
            "target_column": model_run.target_column,
            "task_type": model_run.task_type,
            "model_type": model_run.model_type,
            "metrics": _loads_dict(model_run.metrics_json),
            "top_features": _loads_list(model_run.feature_importance_json)[:5],
            "created_at": model_run.created_at.isoformat(),
        },
    }


def _simulation_step(simulation_run: SimulationRun | None) -> dict[str, Any]:
    if simulation_run is None:
        return _skipped_step(
            "summarize_latest_simulation_run",
            "No simulation runs are saved yet.",
        )

    payload = simulation_run_payload(simulation_run)
    return {
        "name": "summarize_latest_simulation_run",
        "status": "completed",
        "summary": (
            f"Latest simulation run #{simulation_run.id} has yield "
            f"{payload.get('final_yield')} and impurity {payload.get('final_impurity')}."
        ),
        "data": {
            "simulation_run_id": simulation_run.id,
            "simulation_type": simulation_run.simulation_type,
            "input": payload.get("input", {}),
            "result": payload.get("result", {}),
            "created_at": simulation_run.created_at.isoformat(),
        },
    }


def _optimization_step(optimization_run: OptimizationRun | None) -> dict[str, Any]:
    if optimization_run is None:
        return _skipped_step(
            "summarize_latest_optimization_run",
            "No optimization runs are saved yet.",
        )

    payload = optimization_run_payload(optimization_run)
    return {
        "name": "summarize_latest_optimization_run",
        "status": "completed",
        "summary": (
            f"Latest optimization run #{optimization_run.id} found best yield "
            f"{payload.get('best_final_yield')} with impurity "
            f"{payload.get('best_final_impurity')}."
        ),
        "data": {
            "optimization_run_id": optimization_run.id,
            "optimization_type": optimization_run.optimization_type,
            "objective": optimization_run.objective,
            "constraints": payload.get("constraints", {}),
            "best_inputs": payload.get("best_inputs", {}),
            "best_final_yield": payload.get("best_final_yield"),
            "best_final_impurity": payload.get("best_final_impurity"),
            "objective_value": payload.get("objective_value"),
            "created_at": optimization_run.created_at.isoformat(),
        },
    }


def _build_project_analysis_result(
    *,
    project: Project,
    memory: dict[str, Any],
    dataset_step: dict[str, Any],
    document_step: dict[str, Any],
    model_step: dict[str, Any],
    simulation_step: dict[str, Any],
    optimization_step: dict[str, Any],
) -> dict[str, Any]:
    assets = {
        "dataset_available": dataset_step["status"] == "completed",
        "document_context_available": document_step["status"] == "completed",
        "model_available": model_step["status"] == "completed",
        "simulation_available": simulation_step["status"] == "completed",
        "optimization_available": optimization_step["status"] == "completed",
    }
    gaps = _gaps(assets)
    actions = _recommended_actions(assets, memory)
    summary = _status_summary(project, assets, gaps)
    return {
        "project_id": project.id,
        "project_name": project.name,
        "summary": summary,
        "current_assets": assets,
        "gaps": gaps,
        "recommended_next_actions": actions,
    }


def _status_summary(
    project: Project,
    assets: dict[str, bool],
    gaps: list[str],
) -> str:
    available_count = sum(1 for available in assets.values() if available)
    if gaps:
        return (
            f"{project.name} has {available_count} of 5 analysis asset types available. "
            f"The main gap is {gaps[0]}"
        )
    return (
        f"{project.name} has datasets, documents, model results, simulations, "
        "and optimization context available for review."
    )


def _gaps(assets: dict[str, bool]) -> list[str]:
    gaps: list[str] = []
    if not assets["dataset_available"]:
        gaps.append("upload a dataset before modeling or quantitative analysis")
    if not assets["document_context_available"]:
        gaps.append("upload relevant papers, SOPs, or notes for document context")
    if not assets["model_available"]:
        gaps.append("train a baseline model after selecting a target column")
    if not assets["simulation_available"]:
        gaps.append("run a simulation before comparing operating scenarios")
    if not assets["optimization_available"]:
        gaps.append("run an optimization before recommending simulated experiments")
    return gaps


def _recommended_actions(
    assets: dict[str, bool],
    memory: dict[str, Any],
) -> list[str]:
    if not assets["dataset_available"]:
        return [
            "Upload a CSV dataset for the active project.",
            "Add papers, SOPs, or notes that describe the scientific objective.",
            "After upload, choose a target column and train a baseline model.",
        ]
    if not assets["model_available"]:
        target = memory.get("selected_target_column") or "a target column"
        return [
            f"Train a baseline model using {target}.",
            "Inspect missing values and column types before interpreting results.",
            "Ask document questions to connect dataset variables with domain context.",
        ]
    if not assets["simulation_available"]:
        return [
            "Review the latest model metrics and top features for obvious data issues.",
            "Run a simple simulation for the most relevant operating scenario.",
            "Use document context to decide whether the simulation assumptions are plausible.",
        ]
    if not assets["optimization_available"]:
        return [
            "Run a transparent optimization over the saved simulation benchmark.",
            "Compare optimized conditions with the latest simulation result.",
            "Treat simulated recommendations as candidates for expert review, not instructions.",
        ]
    return [
        "Compare the latest optimization against the latest simulation and model evidence.",
        "Review document evidence for constraints or assumptions missing from the model.",
        "Select the next experiment from the top simulated candidates and record the outcome.",
    ]


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


def _list_latest_rows(
    db: Session,
    model: type[Any],
    project_id: int,
) -> list[Any]:
    result = db.execute(
        select(model)
        .where(model.project_id == project_id)
        .order_by(model.created_at.desc(), model.id.desc())
    )
    return list(result.scalars().all())


def _dataset_metadata(dataset: Dataset | None) -> dict[str, Any] | None:
    if dataset is None:
        return None
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
    }


def _skipped_step(name: str, summary: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "skipped",
        "summary": summary,
        "data": None,
    }


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
