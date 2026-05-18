import json
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUploadPreview, DatasetRead
from app.services.id_reset import reset_empty_id_sequences
from app.services.memory_service import delete_memory, update_project_summary, upsert_memory


def create_dataset(
    db: Session,
    project_id: int,
    dataset_data: DatasetCreate,
) -> Dataset:
    dataset = Dataset(
        project_id=project_id,
        filename=dataset_data.filename,
        row_count=dataset_data.row_count,
        column_count=dataset_data.column_count,
        raw_data_json=json.dumps(dataset_data.data),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    dataset_count = _project_dataset_count(db, project_id)
    upsert_memory(
        db,
        project_id,
        "latest_dataset_id",
        dataset.id,
        memory_type="dataset",
        source="dataset_upload",
    )
    upsert_memory(
        db,
        project_id,
        "latest_dataset_filename",
        dataset.filename,
        memory_type="dataset",
        source="dataset_upload",
    )
    upsert_memory(
        db,
        project_id,
        "dataset_count",
        dataset_count,
        memory_type="dataset",
        source="dataset_upload",
    )
    update_project_summary(db, project_id)
    return dataset


def list_project_datasets(db: Session, project_id: int) -> list[Dataset]:
    result = db.execute(
        select(Dataset)
        .where(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
    )
    return list(result.scalars().all())


def get_dataset(db: Session, dataset_id: int) -> Dataset | None:
    return db.get(Dataset, dataset_id)


def delete_dataset(db: Session, dataset: Dataset) -> None:
    project_id = dataset.project_id
    db.delete(dataset)
    db.commit()
    _sync_dataset_memory_after_delete(db, project_id)
    reset_empty_id_sequences(db, [Dataset.__tablename__])
    db.commit()


def _sync_dataset_memory_after_delete(db: Session, project_id: int) -> None:
    datasets = list_project_datasets(db, project_id)
    dataset_count = len(datasets)
    upsert_memory(
        db,
        project_id,
        "dataset_count",
        dataset_count,
        memory_type="dataset",
        source="dataset_delete",
    )
    if datasets:
        latest_dataset = datasets[0]
        upsert_memory(
            db,
            project_id,
            "latest_dataset_id",
            latest_dataset.id,
            memory_type="dataset",
            source="dataset_delete",
        )
        upsert_memory(
            db,
            project_id,
            "latest_dataset_filename",
            latest_dataset.filename,
            memory_type="dataset",
            source="dataset_delete",
        )
        return

    delete_memory(db, project_id, "latest_dataset_id")
    delete_memory(db, project_id, "latest_dataset_filename")
    update_project_summary(db, project_id)


def build_upload_preview(filename: str, dataframe: pd.DataFrame) -> DatasetUploadPreview:
    dataframe = dataframe.copy()
    dataframe.columns = [str(column) for column in dataframe.columns]

    missing_values = {
        column: int(value) for column, value in dataframe.isna().sum().items()
    }
    column_types = {
        column: str(dataframe[column].dtype) for column in dataframe.columns
    }

    clean_dataframe = dataframe.astype(object).where(pd.notna(dataframe), None)
    data: list[dict[str, Any]] = clean_dataframe.to_dict(orient="records")

    return DatasetUploadPreview(
        filename=filename,
        rows=len(dataframe),
        columns=len(dataframe.columns),
        column_names=list(dataframe.columns),
        preview=data[:5],
        data=data,
        profile={
            "missing_values": missing_values,
            "column_types": column_types,
        },
    )


def dataset_to_read_data(dataset: Dataset) -> DatasetRead:
    return DatasetRead(
        id=dataset.id,
        project_id=dataset.project_id,
        filename=dataset.filename,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        data=json.loads(dataset.raw_data_json),
        created_at=dataset.created_at,
    )


def _project_dataset_count(db: Session, project_id: int) -> int:
    result = db.execute(
        select(func.count()).select_from(Dataset).where(Dataset.project_id == project_id)
    )
    return int(result.scalar_one())
