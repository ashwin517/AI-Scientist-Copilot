import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.dataset import DatasetCreate, DatasetRead, DatasetUploadPreview
from app.services.datasets import (
    build_upload_preview,
    create_dataset,
    dataset_to_read_data,
    get_dataset,
    list_project_datasets,
)
from app.services.projects import get_project


router = APIRouter(tags=["datasets"])


@router.post("/datasets/upload-preview", response_model=DatasetUploadPreview)
def upload_dataset_preview_route(
    file: UploadFile = File(...),
) -> DatasetUploadPreview:
    try:
        dataframe = pd.read_csv(file.file)
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file is empty",
        ) from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file could not be parsed",
        ) from exc

    return build_upload_preview(file.filename or "uploaded.csv", dataframe)


@router.post(
    "/projects/{project_id}/datasets",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
)
def create_project_dataset_route(
    project_id: int,
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
) -> DatasetRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    dataset = create_dataset(db, project_id, dataset_data)
    return dataset_to_read_data(dataset)


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetRead])
def list_project_datasets_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> list[DatasetRead]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return [
        dataset_to_read_data(dataset)
        for dataset in list_project_datasets(db, project_id)
    ]


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
def get_dataset_route(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> DatasetRead:
    dataset = get_dataset(db, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    return dataset_to_read_data(dataset)
