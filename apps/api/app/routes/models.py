from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.models import ModelRunRead, ModelTrainingRequest, ModelTrainingResponse
from app.services.datasets import get_dataset
from app.services.model_training import (
    ModelTrainingError,
    list_project_model_runs,
    model_run_to_read_data,
    train_and_persist_baseline_model,
    train_baseline_model,
)
from app.services.projects import get_project


router = APIRouter(prefix="/models", tags=["models"])


@router.post("/train", response_model=ModelTrainingResponse)
def train_model_route(
    training_request: ModelTrainingRequest,
    db: Session = Depends(get_db),
) -> ModelTrainingResponse:
    try:
        if training_request.project_id is not None:
            if get_project(db, training_request.project_id) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Project not found",
                )

            if training_request.dataset_id is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="dataset_id is required for persistent training",
                )

            dataset = get_dataset(db, training_request.dataset_id)
            if dataset is None or dataset.project_id != training_request.project_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found",
                )

            return train_and_persist_baseline_model(
                db,
                training_request.project_id,
                dataset,
                training_request.target_column,
            )

        if training_request.data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training data is required",
            )

        return train_baseline_model(
            training_request.data,
            training_request.target_column,
        )
    except ModelTrainingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("/projects/{project_id}/model-runs", response_model=list[ModelRunRead])
def list_project_model_runs_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> list[ModelRunRead]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return [
        model_run_to_read_data(model_run)
        for model_run in list_project_model_runs(db, project_id)
    ]
