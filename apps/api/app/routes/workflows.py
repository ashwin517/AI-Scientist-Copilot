from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.projects import get_project
from app.workflows.project_analysis_workflow import run_project_analysis_workflow


router = APIRouter(prefix="/projects/{project_id}/workflows", tags=["workflows"])


@router.post("/project-analysis")
def run_project_analysis_workflow_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    try:
        return run_project_analysis_workflow(db, project_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

