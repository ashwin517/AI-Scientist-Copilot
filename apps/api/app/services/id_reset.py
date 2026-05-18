from collections.abc import Iterable

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.db.models import (
    AgentPendingAction,
    ChatMessage,
    Dataset,
    Document,
    DocumentChunk,
    ModelRun,
    Project,
    ProjectMemory,
)


RESETTABLE_TABLES = (
    AgentPendingAction.__tablename__,
    ChatMessage.__tablename__,
    DocumentChunk.__tablename__,
    ModelRun.__tablename__,
    ProjectMemory.__tablename__,
    Dataset.__tablename__,
    Document.__tablename__,
    Project.__tablename__,
)


def reset_empty_id_sequences(
    db: Session,
    table_names: Iterable[str] = RESETTABLE_TABLES,
) -> None:
    bind = db.get_bind()
    dialect_name = bind.dialect.name
    existing_tables = set(inspect(bind).get_table_names())

    for table_name in table_names:
        if table_name not in existing_tables or not _table_is_empty(db, table_name):
            continue

        if dialect_name == "postgresql":
            _reset_postgres_sequence(db, table_name)
        elif dialect_name == "sqlite":
            _reset_sqlite_sequence(db, table_name)


def _table_is_empty(db: Session, table_name: str) -> bool:
    quoted_table = _quote_identifier(table_name)
    result = db.execute(text(f"SELECT 1 FROM {quoted_table} LIMIT 1"))
    return result.first() is None


def _reset_postgres_sequence(db: Session, table_name: str) -> None:
    db.execute(
        text(
            "SELECT setval("
            "pg_get_serial_sequence(:table_name, 'id'), "
            "1, "
            "false"
            ")"
        ),
        {"table_name": table_name},
    )


def _reset_sqlite_sequence(db: Session, table_name: str) -> None:
    if not _sqlite_sequence_table_exists(db):
        return
    db.execute(
        text("DELETE FROM sqlite_sequence WHERE name = :table_name"),
        {"table_name": table_name},
    )


def _sqlite_sequence_table_exists(db: Session) -> bool:
    result = db.execute(
        text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'sqlite_sequence'"
        )
    )
    return result.first() is not None


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
