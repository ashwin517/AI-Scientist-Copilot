import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import WorkflowRun
from app.workflows.schemas import WorkflowRunRead, WorkflowStepRead


def create_workflow_run(
    db: Session,
    project_id: int,
    workflow_type: str,
) -> WorkflowRun:
    workflow_run = WorkflowRun(
        project_id=project_id,
        workflow_type=workflow_type,
        status="running",
        steps_json="[]",
        result_json="{}",
    )
    db.add(workflow_run)
    db.commit()
    db.refresh(workflow_run)
    return workflow_run


def complete_workflow_run(
    db: Session,
    workflow_run: WorkflowRun,
    *,
    steps: list[dict[str, Any]],
    result: dict[str, Any],
    status: str = "completed",
) -> WorkflowRun:
    workflow_run.status = status
    workflow_run.steps_json = json.dumps(steps, sort_keys=True, default=str)
    workflow_run.result_json = json.dumps(result, sort_keys=True, default=str)
    workflow_run.completed_at = datetime.now(timezone.utc)
    db.add(workflow_run)
    db.commit()
    db.refresh(workflow_run)
    return workflow_run


def workflow_run_to_read_data(workflow_run: WorkflowRun) -> WorkflowRunRead:
    return WorkflowRunRead(
        id=workflow_run.id,
        project_id=workflow_run.project_id,
        workflow_type=workflow_run.workflow_type,
        status=workflow_run.status,
        steps=[
            WorkflowStepRead(**step)
            for step in _loads_list(workflow_run.steps_json)
            if isinstance(step, dict)
        ],
        result=_loads_dict(workflow_run.result_json),
        created_at=workflow_run.created_at,
        completed_at=workflow_run.completed_at,
    )


def list_project_workflow_runs(
    db: Session,
    project_id: int,
    limit: int = 10,
) -> list[WorkflowRun]:
    query = (
        select(WorkflowRun)
        .where(WorkflowRun.project_id == project_id)
        .order_by(WorkflowRun.created_at.desc(), WorkflowRun.id.desc())
        .limit(max(1, limit))
    )
    return list(db.execute(query).scalars().all())


def latest_project_workflow_run(
    db: Session,
    project_id: int,
    workflow_run_id: int | None = None,
) -> WorkflowRun | None:
    if workflow_run_id is not None:
        workflow_run = db.get(WorkflowRun, workflow_run_id)
        if workflow_run is not None and workflow_run.project_id == project_id:
            return workflow_run

    runs = list_project_workflow_runs(db, project_id, limit=1)
    return runs[0] if runs else None


def workflow_run_payload(workflow_run: WorkflowRun) -> dict[str, Any]:
    return {
        "workflow_run_id": workflow_run.id,
        "project_id": workflow_run.project_id,
        "workflow_type": workflow_run.workflow_type,
        "status": workflow_run.status,
        "steps": _loads_list(workflow_run.steps_json),
        "result": _loads_dict(workflow_run.result_json),
        "created_at": workflow_run.created_at.isoformat(),
        "completed_at": (
            workflow_run.completed_at.isoformat()
            if workflow_run.completed_at is not None
            else None
        ),
    }


def _loads_dict(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _loads_list(value: str) -> list[Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return decoded if isinstance(decoded, list) else []
