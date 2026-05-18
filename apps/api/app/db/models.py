from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    document_chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    model_runs: Mapped[list["ModelRun"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    pending_actions: Mapped[list["AgentPendingAction"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    memories: Mapped[list["ProjectMemory"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    simulation_runs: Mapped[list["SimulationRun"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    optimization_runs: Mapped[list["OptimizationRun"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    workflow_runs: Mapped[list["WorkflowRun"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    reports: Mapped[list["Report"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_data_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="datasets")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    extracted_text_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    project: Mapped[Project] = relationship(back_populates="document_chunks")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="chat_messages")


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_type: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    feature_importance_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="model_runs")
    dataset: Mapped[Dataset | None] = relationship()


class AgentPendingAction(Base):
    __tablename__ = "agent_pending_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    arguments_json: Mapped[str] = mapped_column(Text, nullable=False)
    missing_fields_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="pending_actions")


class ProjectMemory(Base):
    __tablename__ = "project_memory"
    __table_args__ = (
        UniqueConstraint("project_id", "key", name="uq_project_memory_project_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    memory_type: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="memories")


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    simulation_type: Mapped[str] = mapped_column(String(255), nullable=False)
    input_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="simulation_runs")


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    optimization_type: Mapped[str] = mapped_column(String(255), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    constraints_json: Mapped[str] = mapped_column(Text, nullable=False)
    search_space_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="optimization_runs")


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workflow_type: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    steps_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    project: Mapped[Project] = relationship(back_populates="workflow_runs")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    report_type: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    source_summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    project: Mapped[Project] = relationship(back_populates="reports")


__all__ = [
    "Base",
    "Project",
    "Dataset",
    "Document",
    "DocumentChunk",
    "ChatMessage",
    "ModelRun",
    "AgentPendingAction",
    "ProjectMemory",
    "SimulationRun",
    "OptimizationRun",
    "WorkflowRun",
    "Report",
]
