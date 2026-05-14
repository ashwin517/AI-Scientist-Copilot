import json
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUploadPreview, DatasetRead


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
    db.delete(dataset)
    db.commit()


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
