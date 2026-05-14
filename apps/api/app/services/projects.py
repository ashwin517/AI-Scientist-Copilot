from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Project
from app.schemas.project import ProjectCreate


def create_project(db: Session, project_data: ProjectCreate) -> Project:
    project = Project(**project_data.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def list_projects(db: Session) -> list[Project]:
    result = db.execute(select(Project).order_by(Project.created_at.desc()))
    return list(result.scalars().all())


def get_project(db: Session, project_id: int) -> Project | None:
    return db.get(Project, project_id)
