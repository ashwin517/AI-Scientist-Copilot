import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Dataset, Document, ModelRun, Project, ProjectMemory
from app.schemas.memory import ProjectMemoryRead


MAX_MEMORY_ITEMS_FOR_PROMPT = 20
MAX_MEMORY_VALUE_CHARACTERS = 500
MAX_MEMORY_CONTEXT_CHARACTERS = 2500


def upsert_memory(
    db: Session,
    project_id: int,
    key: str,
    value: Any,
    memory_type: str = "fact",
    source: str | None = None,
) -> ProjectMemory:
    memory = get_memory(db, project_id, key)
    value_json = json.dumps(value, sort_keys=True, default=str)

    if memory is None:
        memory = ProjectMemory(
            project_id=project_id,
            key=key,
            value_json=value_json,
            memory_type=memory_type,
            source=source,
        )
    else:
        memory.value_json = value_json
        memory.memory_type = memory_type
        memory.source = source

    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


def get_memory(db: Session, project_id: int, key: str) -> ProjectMemory | None:
    result = db.execute(
        select(ProjectMemory).where(
            ProjectMemory.project_id == project_id,
            ProjectMemory.key == key,
        )
    )
    return result.scalar_one_or_none()


def list_memory(db: Session, project_id: int) -> list[ProjectMemory]:
    result = db.execute(
        select(ProjectMemory)
        .where(ProjectMemory.project_id == project_id)
        .order_by(ProjectMemory.updated_at.desc(), ProjectMemory.id.desc())
    )
    return list(result.scalars().all())


def delete_memory(db: Session, project_id: int, key: str) -> bool:
    memory = get_memory(db, project_id, key)
    if memory is None:
        return False

    db.delete(memory)
    db.commit()
    return True


def memory_to_read_data(memory: ProjectMemory) -> ProjectMemoryRead:
    return ProjectMemoryRead(
        id=memory.id,
        project_id=memory.project_id,
        memory_type=memory.memory_type,
        key=memory.key,
        value=_decode_value(memory.value_json),
        source=memory.source,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def build_memory_prompt_context(memories: list[ProjectMemory]) -> dict[str, Any]:
    if not memories:
        return {
            "available": False,
            "note": "No project memory has been saved yet.",
        }

    items: list[dict[str, Any]] = []
    for memory in memories[:MAX_MEMORY_ITEMS_FOR_PROMPT]:
        items.append(
            {
                "key": memory.key,
                "type": memory.memory_type,
                "value": _bounded_value(_decode_value(memory.value_json)),
                "source": memory.source,
            }
        )

    memory_values = memory_records_to_dict(memories)
    context: dict[str, Any] = {
        "available": True,
        "project_summary": memory_values.get("project_summary"),
        "items": items,
    }
    if len(memories) > MAX_MEMORY_ITEMS_FOR_PROMPT:
        context["items_omitted"] = len(memories) - MAX_MEMORY_ITEMS_FOR_PROMPT
    return context


def memory_records_to_dict(memories: list[ProjectMemory]) -> dict[str, Any]:
    return {
        memory.key: _decode_value(memory.value_json)
        for memory in memories
    }


def sync_project_memory_from_records(db: Session, project_id: int) -> None:
    _sync_document_memory(db, project_id)
    _sync_dataset_memory(db, project_id)
    _sync_model_memory(db, project_id)


def update_project_summary(db: Session, project_id: int) -> str:
    sync_project_memory_from_records(db, project_id)
    project = db.get(Project, project_id)
    memories = memory_records_to_dict(list_memory(db, project_id))
    latest_model_run = _latest_project_row(db, ModelRun, project_id)

    summary = _build_project_summary_text(
        project=project,
        memories=memories,
        latest_model_run=latest_model_run,
    )
    upsert_memory(
        db,
        project_id,
        "project_summary",
        summary,
        memory_type="summary",
        source="memory_summary",
    )
    return summary


def _sync_document_memory(db: Session, project_id: int) -> None:
    count = _count_project_rows(db, Document, project_id)
    upsert_memory(
        db,
        project_id,
        "document_count",
        count,
        memory_type="document",
        source="memory_sync",
    )
    latest_document = _latest_project_row(db, Document, project_id)
    if latest_document is None:
        delete_memory(db, project_id, "latest_document_id")
        delete_memory(db, project_id, "latest_document_filename")
        return

    upsert_memory(
        db,
        project_id,
        "latest_document_id",
        latest_document.id,
        memory_type="document",
        source="memory_sync",
    )
    upsert_memory(
        db,
        project_id,
        "latest_document_filename",
        latest_document.filename,
        memory_type="document",
        source="memory_sync",
    )


def _sync_dataset_memory(db: Session, project_id: int) -> None:
    count = _count_project_rows(db, Dataset, project_id)
    upsert_memory(
        db,
        project_id,
        "dataset_count",
        count,
        memory_type="dataset",
        source="memory_sync",
    )
    remembered_dataset = _remembered_project_dataset(db, project_id)
    dataset = remembered_dataset or _latest_project_row(db, Dataset, project_id)
    if dataset is None:
        delete_memory(db, project_id, "latest_dataset_id")
        delete_memory(db, project_id, "latest_dataset_filename")
        return

    upsert_memory(
        db,
        project_id,
        "latest_dataset_id",
        dataset.id,
        memory_type="dataset",
        source="memory_sync",
    )
    upsert_memory(
        db,
        project_id,
        "latest_dataset_filename",
        dataset.filename,
        memory_type="dataset",
        source="memory_sync",
    )


def _sync_model_memory(db: Session, project_id: int) -> None:
    latest_model_run = _latest_project_row(db, ModelRun, project_id)
    if latest_model_run is None:
        delete_memory(db, project_id, "latest_model_run_id")
        delete_memory(db, project_id, "latest_task_type")
        return

    upsert_memory(
        db,
        project_id,
        "latest_model_run_id",
        latest_model_run.id,
        memory_type="model",
        source="memory_sync",
    )
    upsert_memory(
        db,
        project_id,
        "selected_target_column",
        latest_model_run.target_column,
        memory_type="model",
        source="memory_sync",
    )
    upsert_memory(
        db,
        project_id,
        "latest_task_type",
        latest_model_run.task_type,
        memory_type="model",
        source="memory_sync",
    )


def _count_project_rows(db: Session, model: type[Any], project_id: int) -> int:
    result = db.execute(
        select(func.count()).select_from(model).where(model.project_id == project_id)
    )
    return int(result.scalar_one())


def _latest_project_row(
    db: Session,
    model: type[Any],
    project_id: int,
) -> Any | None:
    result = db.execute(
        select(model)
        .where(model.project_id == project_id)
        .order_by(model.created_at.desc(), model.id.desc())
    )
    return result.scalars().first()


def _remembered_project_dataset(db: Session, project_id: int) -> Dataset | None:
    memory = get_memory(db, project_id, "latest_dataset_id")
    if memory is None:
        return None
    dataset_id = _as_int(_decode_value(memory.value_json))
    if not dataset_id:
        return None
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.project_id != project_id:
        return None
    return dataset


def _build_project_summary_text(
    project: Project | None,
    memories: dict[str, Any],
    latest_model_run: ModelRun | None,
) -> str:
    parts: list[str] = []

    user_facts = _user_project_facts(memories)
    document_count = _as_int(memories.get("document_count"))
    dataset_count = _as_int(memories.get("dataset_count"))
    selected_target = memories.get("selected_target_column")
    has_summary_context = bool(
        user_facts
        or document_count
        or dataset_count
        or selected_target
        or latest_model_run is not None
        or (project is not None and project.description)
    )
    if not has_summary_context:
        return "No project summary has been created yet."

    if project is not None:
        parts.append(f"This project is named {project.name}.")
        if project.description:
            parts.append(f"Project description: {project.description}.")

    if user_facts:
        parts.extend(user_facts)

    latest_document = memories.get("latest_document_filename")
    if document_count:
        if latest_document:
            parts.append(
                f"It contains {document_count} uploaded document"
                f"{_plural_suffix(document_count)}; the latest is {latest_document}."
            )
        else:
            parts.append(
                f"It contains {document_count} uploaded document"
                f"{_plural_suffix(document_count)}."
            )

    latest_dataset = memories.get("latest_dataset_filename")
    if dataset_count:
        if latest_dataset:
            parts.append(
                f"It has {dataset_count} dataset{_plural_suffix(dataset_count)}; "
                f"the latest is {latest_dataset}."
            )
        else:
            parts.append(f"It has {dataset_count} dataset{_plural_suffix(dataset_count)}.")

    if selected_target:
        parts.append(f"The selected target column is {selected_target}.")

    if latest_model_run is not None:
        parts.append(
            f"The latest model run is #{latest_model_run.id}, a "
            f"{latest_model_run.model_type} {latest_model_run.task_type} model."
        )

    return " ".join(parts)


def _user_project_facts(memories: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    for key in ("project_domain_note", "project_note"):
        value = memories.get(key)
        if isinstance(value, str) and value:
            facts.append(value.rstrip(".") + ".")
    return facts


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _plural_suffix(count: int) -> str:
    return "" if count == 1 else "s"


def _decode_value(value_json: str) -> Any:
    try:
        return json.loads(value_json)
    except json.JSONDecodeError:
        return value_json


def _bounded_value(value: Any) -> Any:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) <= MAX_MEMORY_VALUE_CHARACTERS:
        return value
    omitted = len(text) - MAX_MEMORY_VALUE_CHARACTERS
    return text[:MAX_MEMORY_VALUE_CHARACTERS].rstrip() + (
        f"...[truncated {omitted} characters]"
    )
