from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.memory import ProjectMemoryRead
from app.services.memory_service import (
    delete_memory,
    list_memory,
    memory_to_read_data,
    update_project_summary,
)
from app.services.projects import get_project


router = APIRouter(tags=["memory"])


@router.get("/projects/{project_id}/memory", response_model=list[ProjectMemoryRead])
def list_project_memory_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> list[ProjectMemoryRead]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    update_project_summary(db, project_id)
    return [
        memory_to_read_data(memory)
        for memory in list_memory(db, project_id)
    ]


@router.delete(
    "/projects/{project_id}/memory/{key}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_project_memory_route(
    project_id: int,
    key: str,
    db: Session = Depends(get_db),
) -> None:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    if not delete_memory(db, project_id, key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory key not found",
        )
