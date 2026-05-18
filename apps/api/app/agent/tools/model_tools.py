import json
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.agent.tools.dataset_tools import resolve_dataset_with_source
from app.db.models import Dataset, ModelRun
from app.services.model_training import (
    ModelTrainingError,
    list_project_model_runs,
    model_run_to_read_data,
    train_and_persist_baseline_model,
)


MAX_EXPLANATION_FEATURES = 8
NO_MODEL_AVAILABLE_MESSAGE = "No trained model is available yet. Train a model first."


def list_model_runs(
    db: Session,
    project_id: int,
    model_run_id: int | None = None,
    project_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    used_memory = False
    if model_run_id is None:
        model_run_id = _remembered_int(project_memory, "latest_model_run_id")
        used_memory = model_run_id is not None

    model_runs = list_project_model_runs(db, project_id)
    if model_run_id is not None:
        selected = [
            model_run
            for model_run in model_runs
            if model_run.id == model_run_id
        ]
        if selected:
            model_runs = selected
        elif used_memory:
            used_memory = False

    memory_notes = []
    if used_memory and model_runs:
        memory_notes.append(
            f"I used your latest model run: #{model_runs[0].id}."
        )

    return {
        "model_runs": [
            model_run_to_read_data(model_run).model_dump(mode="json")
            for model_run in model_runs
        ],
        "count": len(model_runs),
        "selected_model_run_id": (
            model_runs[0].id if model_run_id and model_runs else None
        ),
        "memory_notes": memory_notes,
    }


def explain_latest_model(
    db: Session,
    project_id: int,
    project_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_run = _resolve_latest_model_run(db, project_id, project_memory)
    if model_run is None:
        return {
            "message": NO_MODEL_AVAILABLE_MESSAGE,
            "model_available": False,
            "limitations": [],
            "suggested_next_steps": ["Train a model first."],
        }

    dataset = db.get(Dataset, model_run.dataset_id) if model_run.dataset_id else None
    metrics = _decode_json_dict(model_run.metrics_json)
    top_features = _decode_feature_importance(model_run.feature_importance_json)
    limitations = [
        "Feature importance indicates predictive association, not necessarily causation.",
        "This baseline model should be validated with holdout data, domain review, and error analysis before operational use.",
    ]

    return {
        "message": None,
        "model_available": True,
        "model_run_id": model_run.id,
        "dataset_id": model_run.dataset_id,
        "dataset": _dataset_payload(dataset),
        "target_column": model_run.target_column,
        "task_type": model_run.task_type,
        "model_type": model_run.model_type,
        "metrics": metrics,
        "top_features": top_features,
        "limitations": limitations,
        "suggested_next_steps": _suggest_next_steps(model_run.task_type, metrics),
    }


def train_baseline_model(
    db: Session,
    project_id: int,
    dataset_id: int | None = None,
    target_column: str | None = None,
    latest: bool = True,
    project_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset_id is None and not latest:
        raise ValueError("Please specify a dataset_id or ask to use the latest dataset.")

    try:
        resolved_dataset = resolve_dataset_with_source(
            db,
            project_id,
            dataset_id,
            latest,
            project_memory,
        )
        dataset = resolved_dataset.dataset
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

    available_columns = _available_columns(rows)
    available_columns_text = _format_available_columns(available_columns)
    used_target_memory = False

    if not target_column:
        remembered_target = _remembered_text(
            project_memory,
            "selected_target_column",
        )
        if remembered_target:
            target_column = remembered_target
            used_target_memory = True

    if not target_column:
        raise ValueError(
            f"Please specify a target column. Available columns are: {available_columns_text}"
        )

    if target_column not in available_columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in this dataset. "
            f"Available columns are: {available_columns_text}"
        )

    try:
        result = train_and_persist_baseline_model(
            db,
            project_id,
            dataset,
            target_column,
        )
    except ModelTrainingError as exc:
        raise ValueError(str(exc)) from exc

    memory_notes = []
    if resolved_dataset.used_memory:
        memory_notes.append(
            f"I used your latest uploaded dataset: {dataset.filename}."
        )
    if used_target_memory:
        memory_notes.append(
            f"I used your remembered target column: {target_column}."
        )

    return {
        "model_run_id": result.model_run_id,
        "dataset_id": dataset.id,
        "target_column": target_column,
        "task_type": result.task_type,
        "metrics": result.metrics,
        "feature_importance": [
            item.model_dump() for item in result.feature_importance[:10]
        ],
        "dataset": {
            "id": dataset.id,
            "filename": dataset.filename,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
        },
        "model_result": result.model_dump(),
        "memory_notes": memory_notes,
    }


def _resolve_latest_model_run(
    db: Session,
    project_id: int,
    project_memory: dict[str, Any] | None,
) -> ModelRun | None:
    remembered_model_run_id = _remembered_int(project_memory, "latest_model_run_id")
    if remembered_model_run_id is not None:
        model_run = db.get(ModelRun, remembered_model_run_id)
        if model_run is not None and model_run.project_id == project_id:
            return model_run

    model_runs = list_project_model_runs(db, project_id)
    return model_runs[0] if model_runs else None


def _decode_json_dict(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _decode_feature_importance(value: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    features = [item for item in decoded if isinstance(item, dict)]
    return features[:MAX_EXPLANATION_FEATURES]


def _dataset_payload(dataset: Dataset | None) -> dict[str, Any] | None:
    if dataset is None:
        return None
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
    }


def _suggest_next_steps(task_type: str, metrics: dict[str, Any]) -> list[str]:
    steps = [
        "Review prediction errors by segment or operating condition.",
        "Check for leakage, missing-value patterns, and unstable feature definitions.",
        "Compare against a simple interpretable baseline before adding complexity.",
    ]
    if task_type == "regression":
        steps.append("Inspect residuals and consider MAE/RMSE against domain tolerances.")
    elif task_type == "classification":
        steps.append("Inspect confusion matrix, class balance, precision, and recall.")
    if not metrics:
        steps.append("Recompute or inspect metrics because no metric values were saved.")
    return steps


def _available_columns(rows: list[Any]) -> list[str]:
    if not rows:
        return []
    dataframe = pd.DataFrame(rows)
    return [str(column) for column in dataframe.columns]


def _format_available_columns(columns: list[str]) -> str:
    if not columns:
        return "(none)"
    return ", ".join(columns)


def _remembered_int(
    project_memory: dict[str, Any] | None,
    key: str,
) -> int | None:
    if not project_memory:
        return None
    value = project_memory.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _remembered_text(
    project_memory: dict[str, Any] | None,
    key: str,
) -> str | None:
    if not project_memory:
        return None
    value = project_memory.get(key)
    if isinstance(value, str) and value:
        return value
    return None
