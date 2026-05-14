from fastapi import APIRouter, HTTPException, status

from app.schemas.models import ModelTrainingRequest, ModelTrainingResponse
from app.services.model_training import ModelTrainingError, train_baseline_model


router = APIRouter(prefix="/models", tags=["models"])


@router.post("/train", response_model=ModelTrainingResponse)
def train_model_route(
    training_request: ModelTrainingRequest,
) -> ModelTrainingResponse:
    try:
        return train_baseline_model(
            training_request.data,
            training_request.target_column,
        )
    except ModelTrainingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
