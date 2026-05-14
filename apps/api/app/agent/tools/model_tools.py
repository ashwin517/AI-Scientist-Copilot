import json
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.agent.tools.dataset_tools import resolve_dataset
from app.services.model_training import (
    ModelTrainingError,
    list_project_model_runs,
    model_run_to_read_data,
    train_and_persist_baseline_model,
)


def list_model_runs(db: Session, project_id: int) -> dict[str, Any]:
    model_runs = list_project_model_runs(db, project_id)
    return {
        "model_runs": [
            model_run_to_read_data(model_run).model_dump(mode="json")
            for model_run in model_runs
        ],
        "count": len(model_runs),
    }


def train_baseline_model(
    db: Session,
    project_id: int,
    dataset_id: int | None = None,
    target_column: str | None = None,
    latest: bool = True,
) -> dict[str, Any]:
    if dataset_id is None and not latest:
        raise ValueError("Please specify a dataset_id or ask to use the latest dataset.")

    try:
        dataset = resolve_dataset(db, project_id, dataset_id, latest)
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
    }


def _available_columns(rows: list[Any]) -> list[str]:
    if not rows:
        return []
    dataframe = pd.DataFrame(rows)
    return [str(column) for column in dataframe.columns]


def _format_available_columns(columns: list[str]) -> str:
    if not columns:
        return "(none)"
    return ", ".join(columns)
