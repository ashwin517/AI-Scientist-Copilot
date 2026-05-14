import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AgentPendingAction


def get_pending_action(
    db: Session,
    project_id: int,
) -> AgentPendingAction | None:
    result = db.execute(
        select(AgentPendingAction)
        .where(AgentPendingAction.project_id == project_id)
        .order_by(AgentPendingAction.updated_at.desc(), AgentPendingAction.id.desc())
    )
    return result.scalars().first()


def save_pending_action(
    db: Session,
    project_id: int,
    tool_name: str,
    arguments: dict[str, Any],
    missing_fields: list[str],
) -> AgentPendingAction:
    clear_pending_action(db, project_id)
    pending_action = AgentPendingAction(
        project_id=project_id,
        tool_name=tool_name,
        arguments_json=json.dumps(arguments),
        missing_fields_json=json.dumps(missing_fields),
    )
    db.add(pending_action)
    db.commit()
    db.refresh(pending_action)
    return pending_action


def update_pending_action(
    db: Session,
    pending_action: AgentPendingAction,
    arguments: dict[str, Any],
    missing_fields: list[str],
) -> AgentPendingAction:
    pending_action.arguments_json = json.dumps(arguments)
    pending_action.missing_fields_json = json.dumps(missing_fields)
    db.add(pending_action)
    db.commit()
    db.refresh(pending_action)
    return pending_action


def clear_pending_action(db: Session, project_id: int) -> None:
    pending_action = get_pending_action(db, project_id)
    if pending_action is None:
        return
    db.delete(pending_action)
    db.commit()


def pending_arguments(pending_action: AgentPendingAction) -> dict[str, Any]:
    value = json.loads(pending_action.arguments_json)
    return value if isinstance(value, dict) else {}


def pending_missing_fields(pending_action: AgentPendingAction) -> list[str]:
    value = json.loads(pending_action.missing_fields_json)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]

