from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.projects import get_project
from app.simulation.schemas import BatchReactorSimulationInput
from app.simulation.simulation_service import (
    run_and_persist_batch_reactor_simulation,
    simulation_run_payload,
)


router = APIRouter(prefix="/projects/{project_id}/simulation", tags=["simulation"])


@router.post("/batch-reactor")
def run_batch_reactor_simulation_route(
    project_id: int,
    simulation_input: BatchReactorSimulationInput,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    try:
        simulation_run, result = run_and_persist_batch_reactor_simulation(
            db,
            project_id,
            simulation_input,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return simulation_run_payload(simulation_run, result)
