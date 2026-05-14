from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.base import Base


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    from app.db import models

    _ = models
    Base.metadata.create_all(bind=engine)
    sync_dev_schema()


def sync_dev_schema() -> None:
    inspector = inspect(engine)
    if "datasets" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("datasets")}

    with engine.begin() as connection:
        if "name" in columns and "filename" not in columns:
            connection.execute(
                text("ALTER TABLE datasets RENAME COLUMN name TO filename")
            )
            columns.remove("name")
            columns.add("filename")

        if "filename" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE datasets "
                    "ADD COLUMN filename VARCHAR(255) NOT NULL DEFAULT 'untitled-dataset'"
                )
            )

        if "row_count" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE datasets "
                    "ADD COLUMN row_count INTEGER NOT NULL DEFAULT 0"
                )
            )

        if "column_count" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE datasets "
                    "ADD COLUMN column_count INTEGER NOT NULL DEFAULT 0"
                )
            )

        if "raw_data_json" not in columns:
            connection.execute(
                text(
                    "ALTER TABLE datasets "
                    "ADD COLUMN raw_data_json TEXT NOT NULL DEFAULT '[]'"
                )
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
