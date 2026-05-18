import json
from typing import Any

from sqlalchemy.orm import Session

from app.services.memory_service import get_memory
from app.workflows.project_analysis_workflow import run_project_analysis_workflow
from app.workflows.workflow_service import (
    latest_project_workflow_run,
    list_project_workflow_runs,
    workflow_run_payload,
)


def run_project_analysis_workflow_tool(
    db: Session,
    project_id: int,
) -> dict[str, Any]:
    return run_project_analysis_workflow(db, project_id)


def list_workflow_runs(
    db: Session,
    project_id: int,
    limit: int = 10,
) -> dict[str, Any]:
    workflow_runs = list_project_workflow_runs(db, project_id, limit=limit)
    return {
        "workflow_runs": [
            _workflow_summary_payload(workflow_run_payload(workflow_run))
            for workflow_run in workflow_runs
        ],
        "count": len(workflow_runs),
    }


def explain_latest_workflow(
    db: Session,
    project_id: int,
) -> dict[str, Any]:
    workflow_run = latest_project_workflow_run(
        db,
        project_id,
        _remembered_latest_workflow_run_id(db, project_id),
    )
    if workflow_run is None:
        return {
            "workflow_available": False,
            "message": "No workflow runs exist for this project yet.",
        }

    payload = workflow_run_payload(workflow_run)
    result = _dict_value(payload.get("result"))
    return {
        "workflow_available": True,
        **_workflow_summary_payload(payload),
        "steps": payload.get("steps", []),
        "summary": result.get("summary"),
        "current_assets": _dict_value(result.get("current_assets")),
        "gaps": _list_value(result.get("gaps")),
        "recommended_next_actions": _list_value(
            result.get("recommended_next_actions")
        ),
    }


def compare_workflow_runs(
    db: Session,
    project_id: int,
) -> dict[str, Any]:
    workflow_runs = list_project_workflow_runs(db, project_id, limit=2)
    if not workflow_runs:
        return {
            "comparison_available": False,
            "message": "No workflow runs exist for this project yet.",
        }
    if len(workflow_runs) < 2:
        return {
            "comparison_available": False,
            "message": "Only one workflow run exists, so there is nothing to compare yet.",
            "latest_workflow": _workflow_summary_payload(
                workflow_run_payload(workflow_runs[0])
            ),
        }

    latest = workflow_run_payload(workflow_runs[0])
    previous = workflow_run_payload(workflow_runs[1])
    latest_result = _dict_value(latest.get("result"))
    previous_result = _dict_value(previous.get("result"))
    latest_assets = _dict_value(latest_result.get("current_assets"))
    previous_assets = _dict_value(previous_result.get("current_assets"))
    latest_gaps = _string_list(latest_result.get("gaps"))
    previous_gaps = _string_list(previous_result.get("gaps"))
    latest_actions = _string_list(latest_result.get("recommended_next_actions"))
    previous_actions = _string_list(previous_result.get("recommended_next_actions"))

    return {
        "comparison_available": True,
        "latest_workflow": _workflow_summary_payload(latest),
        "previous_workflow": _workflow_summary_payload(previous),
        "status_comparison": {
            "latest_status": latest.get("status"),
            "previous_status": previous.get("status"),
            "changed": latest.get("status") != previous.get("status"),
        },
        "major_project_changes": _asset_changes(previous_assets, latest_assets),
        "gaps": {
            "latest": latest_gaps,
            "previous": previous_gaps,
            "new": _ordered_difference(latest_gaps, previous_gaps),
            "resolved": _ordered_difference(previous_gaps, latest_gaps),
        },
        "recommendations": {
            "latest": latest_actions,
            "previous": previous_actions,
            "new": _ordered_difference(latest_actions, previous_actions),
            "removed": _ordered_difference(previous_actions, latest_actions),
        },
    }


def _workflow_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = _dict_value(payload.get("result"))
    return {
        "workflow_run_id": payload.get("workflow_run_id"),
        "workflow_type": payload.get("workflow_type"),
        "status": payload.get("status"),
        "created_at": payload.get("created_at"),
        "completed_at": payload.get("completed_at"),
        "summary": result.get("summary"),
        "current_assets": _dict_value(result.get("current_assets")),
        "gaps": _list_value(result.get("gaps")),
        "recommended_next_actions": _list_value(
            result.get("recommended_next_actions")
        ),
    }


def _remembered_latest_workflow_run_id(db: Session, project_id: int) -> int | None:
    memory = get_memory(db, project_id, "latest_workflow_run_id")
    if memory is None:
        return None
    return _as_int(_decode_memory(memory.value_json))


def _asset_changes(
    previous_assets: dict[str, Any],
    latest_assets: dict[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for key in sorted(set(previous_assets) | set(latest_assets)):
        previous = bool(previous_assets.get(key))
        latest = bool(latest_assets.get(key))
        if previous == latest:
            continue
        changes.append(
            {
                "asset": key,
                "previous": previous,
                "latest": latest,
                "change": "became_available" if latest else "became_missing",
            }
        )
    return changes


def _ordered_difference(values: list[str], comparison: list[str]) -> list[str]:
    comparison_set = set(comparison)
    return [value for value in values if value not in comparison_set]


def _decode_memory(value_json: str) -> Any:
    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        return value_json


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
