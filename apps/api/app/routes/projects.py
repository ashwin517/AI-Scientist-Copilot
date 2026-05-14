from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.project import ProjectCreate, ProjectRead
from app.services.projects import (
    create_project,
    delete_project,
    get_project,
    list_projects,
)


router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project_route(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
) -> ProjectRead:
    return create_project(db, project_data)


@router.get("", response_model=list[ProjectRead])
def list_projects_route(db: Session = Depends(get_db)) -> list[ProjectRead]:
    return list_projects(db)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> ProjectRead:
    project = get_project(db, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> None:
    project = get_project(db, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    delete_project(db, project)
