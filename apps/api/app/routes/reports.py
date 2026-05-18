from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.reports.report_service import (
    generate_project_report,
    get_project_report,
    list_project_reports,
    report_to_list_item,
    report_to_read_data,
)
from app.reports.schemas import ReportListItem, ReportRead
from app.services.projects import get_project


router = APIRouter(prefix="/projects/{project_id}/reports", tags=["reports"])


@router.post("/generate", response_model=ReportRead)
def generate_project_report_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> ReportRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    try:
        return report_to_read_data(generate_project_report(db, project_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[ReportListItem])
def list_project_reports_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> list[ReportListItem]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return [
        report_to_list_item(report)
        for report in list_project_reports(db, project_id)
    ]


@router.get("/{report_id}", response_model=ReportRead)
def get_project_report_route(
    project_id: int,
    report_id: int,
    db: Session = Depends(get_db),
) -> ReportRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    report = get_project_report(db, project_id, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )
    return report_to_read_data(report)

