import json
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.services.datasets import get_dataset, list_project_datasets


MAX_PREVIEW_ROWS = 5
MAX_COLUMNS = 80


def list_datasets(db: Session, project_id: int) -> dict[str, Any]:
    datasets = list_project_datasets(db, project_id)
    return {
        "project_id": project_id,
        "datasets": [
            {
                "id": dataset.id,
                "filename": dataset.filename,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "created_at": dataset.created_at.isoformat(),
            }
            for dataset in datasets
        ],
        "count": len(datasets),
        "note": (
            "Datasets are listed only for the active project. Upload preview files "
            "are not included until they are saved to the project."
        ),
    }


def get_dataset_summary(
    db: Session,
    project_id: int,
    dataset_id: int | None = None,
    latest: bool = True,
) -> dict[str, Any]:
    dataset = resolve_dataset(db, project_id, dataset_id, latest)
    dataframe = _dataset_to_dataframe(dataset)
    missing_values = {
        column: int(value) for column, value in dataframe.isna().sum().items()
    }
    column_types = {
        str(column): str(dataframe[column].dtype) for column in dataframe.columns
    }
    preview = (
        dataframe.head(MAX_PREVIEW_ROWS)
        .astype(object)
        .where(pd.notna(dataframe.head(MAX_PREVIEW_ROWS)), None)
        .to_dict(orient="records")
    )
    return {
        "dataset": _dataset_metadata(dataset),
        "column_names": [str(column) for column in dataframe.columns[:MAX_COLUMNS]],
        "column_types": column_types,
        "missing_values": missing_values,
        "preview": preview,
    }


def show_missing_values(
    db: Session,
    project_id: int,
    dataset_id: int | None = None,
    latest: bool = True,
) -> dict[str, Any]:
    dataset = resolve_dataset(db, project_id, dataset_id, latest)
    dataframe = _dataset_to_dataframe(dataset)
    missing_values = {
        str(column): int(value) for column, value in dataframe.isna().sum().items()
    }
    columns_with_missing = {
        column: count for column, count in missing_values.items() if count > 0
    }
    return {
        "dataset": _dataset_metadata(dataset),
        "missing_values": missing_values,
        "columns_with_missing": columns_with_missing,
        "total_missing_values": sum(missing_values.values()),
    }


def resolve_dataset(
    db: Session,
    project_id: int,
    dataset_id: int | None,
    latest: bool,
) -> Dataset:
    if dataset_id is not None:
        dataset = get_dataset(db, dataset_id)
        if dataset is None or dataset.project_id != project_id:
            raise ValueError("Dataset not found for this project.")
        return dataset

    if latest:
        datasets = list_project_datasets(db, project_id)
        if datasets:
            return datasets[0]

    raise ValueError("No dataset is available for this project.")


def _dataset_to_dataframe(dataset: Dataset) -> pd.DataFrame:
    try:
        rows = json.loads(dataset.raw_data_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Saved dataset could not be decoded.") from exc
    if not isinstance(rows, list):
        raise ValueError("Saved dataset has an invalid format.")
    return pd.DataFrame(rows)


def _dataset_metadata(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "created_at": dataset.created_at.isoformat(),
    }
