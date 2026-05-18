from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.optimization.optimization_service import (
    optimization_run_payload,
    run_and_persist_batch_reactor_optimization,
)
from app.optimization.schemas import BatchReactorOptimizationInput
from app.services.projects import get_project


router = APIRouter(
    prefix="/projects/{project_id}/optimization",
    tags=["optimization"],
)


@router.post("/batch-reactor")
def run_batch_reactor_optimization_route(
    project_id: int,
    optimization_input: BatchReactorOptimizationInput | None = None,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    try:
        optimization_run, result = run_and_persist_batch_reactor_optimization(
            db,
            project_id,
            optimization_input or BatchReactorOptimizationInput(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return optimization_run_payload(optimization_run, result)

