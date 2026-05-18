import json
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.agent.tools.dataset_tools import resolve_dataset_with_source
from app.db.models import Dataset
from app.services.datasets import list_project_datasets
from app.services.memory_service import (
    delete_memory,
    list_memory,
    memory_to_read_data,
    update_project_summary,
    upsert_memory,
)


FORGET_KEY_ALIASES = {
    "target": "selected_target_column",
    "target column": "selected_target_column",
    "selected target": "selected_target_column",
    "selected target column": "selected_target_column",
    "dataset": "latest_dataset_id",
    "latest dataset": "latest_dataset_id",
    "document": "latest_document_id",
    "latest document": "latest_document_id",
    "model": "latest_model_run_id",
    "latest model": "latest_model_run_id",
    "model run": "latest_model_run_id",
    "latest model run": "latest_model_run_id",
    "project domain": "project_domain_note",
    "domain": "project_domain_note",
}


def list_project_memory(db: Session, project_id: int) -> dict[str, Any]:
    update_project_summary(db, project_id)
    memories = list_memory(db, project_id)
    return {
        "memories": [
            memory_to_read_data(memory).model_dump(mode="json")
            for memory in memories
        ],
        "count": len(memories),
    }


def upsert_project_memory(
    db: Session,
    project_id: int,
    key: str,
    value: Any,
    memory_type: str = "fact",
    source: str = "user_instruction",
    validate_target_column: bool = False,
    project_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    matched_dataset: Dataset | None = None
    if validate_target_column:
        value = str(value)
        validation = _validate_target_column(
            db,
            project_id,
            value,
            project_memory,
        )
        if validation["status"] != "valid":
            return {
                "key": key,
                "value": value,
                "updated": False,
                "error": validation["error"],
                "available_columns": validation["available_columns"],
                "matching_datasets": validation["matching_datasets"],
            }
        matched_dataset = validation["dataset"]

    memory = upsert_memory(
        db,
        project_id,
        key,
        value,
        memory_type=memory_type,
        source=source,
    )
    matched_dataset_payload = None
    if matched_dataset is not None:
        _remember_modeling_dataset(db, project_id, matched_dataset)
        matched_dataset_payload = {
            "id": matched_dataset.id,
            "filename": matched_dataset.filename,
        }
    update_project_summary(db, project_id)
    return {
        "key": memory.key,
        "value": memory_to_read_data(memory).value,
        "memory_type": memory.memory_type,
        "source": memory.source,
        "updated": True,
        "matched_dataset": matched_dataset_payload,
    }


def delete_project_memory(
    db: Session,
    project_id: int,
    key: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    resolved_key = _resolve_forget_key(key, label)
    if resolved_key is None:
        return {
            "deleted": False,
            "needs_clarification": True,
            "message": (
                "Which project memory should I forget? For example: target column, "
                "latest dataset, latest model run, or project domain."
            ),
        }

    deleted = delete_memory(db, project_id, resolved_key)
    update_project_summary(db, project_id)
    return {
        "key": resolved_key,
        "deleted": deleted,
    }


def _validate_target_column(
    db: Session,
    project_id: int,
    target_column: str,
    project_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    active_dataset: Dataset | None = None
    active_columns: list[str] = []
    try:
        resolved = resolve_dataset_with_source(
            db,
            project_id,
            dataset_id=None,
            latest=True,
            project_memory=project_memory,
        )
        active_dataset = resolved.dataset
        active_columns = _dataset_columns(active_dataset)
    except ValueError:
        pass

    if target_column in active_columns and active_dataset is not None:
        return {
            "status": "valid",
            "dataset": active_dataset,
            "available_columns": active_columns,
            "matching_datasets": [
                _dataset_payload(active_dataset),
            ],
            "error": None,
        }

    matching_datasets: list[Dataset] = []
    all_columns: list[str] = []
    for dataset in list_project_datasets(db, project_id):
        columns = _dataset_columns(dataset)
        for column in columns:
            if column not in all_columns:
                all_columns.append(column)
        if target_column in columns:
            matching_datasets.append(dataset)

    if len(matching_datasets) == 1:
        return {
            "status": "valid",
            "dataset": matching_datasets[0],
            "available_columns": _dataset_columns(matching_datasets[0]),
            "matching_datasets": [_dataset_payload(matching_datasets[0])],
            "error": None,
        }

    if len(matching_datasets) > 1:
        return {
            "status": "ambiguous",
            "dataset": None,
            "available_columns": all_columns,
            "matching_datasets": [
                _dataset_payload(dataset) for dataset in matching_datasets
            ],
            "error": (
                f"Target column '{target_column}' exists in multiple datasets. "
                "Please specify which dataset to use."
            ),
        }

    return {
        "status": "missing",
        "dataset": None,
        "available_columns": all_columns or active_columns,
        "matching_datasets": [],
        "error": (
            f"Target column '{target_column}' was not found in any saved project dataset."
        ),
    }


def _dataset_columns(dataset: Dataset) -> list[str]:
    try:
        rows = json.loads(dataset.raw_data_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Saved dataset could not be decoded.") from exc
    if not isinstance(rows, list):
        raise ValueError("Saved dataset has an invalid format.")
    if not rows:
        return []
    dataframe = pd.DataFrame(rows)
    return [str(column) for column in dataframe.columns]


def _remember_modeling_dataset(
    db: Session,
    project_id: int,
    dataset: Dataset,
) -> None:
    upsert_memory(
        db,
        project_id,
        "latest_dataset_id",
        dataset.id,
        memory_type="dataset",
        source="target_column_selection",
    )
    upsert_memory(
        db,
        project_id,
        "latest_dataset_filename",
        dataset.filename,
        memory_type="dataset",
        source="target_column_selection",
    )


def _dataset_payload(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
    }


def _resolve_forget_key(key: str | None, label: str | None) -> str | None:
    if key:
        return key
    if not label:
        return None
    normalized = label.strip().casefold()
    if not normalized:
        return None
    if normalized in FORGET_KEY_ALIASES:
        return FORGET_KEY_ALIASES[normalized]
    if normalized in {
        "memory",
        "something",
        "that",
        "this",
        "it",
        "everything",
        "all",
    }:
        return None
    return normalized.replace(" ", "_")
