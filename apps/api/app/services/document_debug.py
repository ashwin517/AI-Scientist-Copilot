from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk


@dataclass(frozen=True)
class DocumentDebugItem:
    id: int
    filename: str
    has_extracted_text: bool
    chunk_count: int
    embedded_chunk_count: int


@dataclass(frozen=True)
class ProjectDocumentDebug:
    documents_count: int
    chunks_count: int
    embedded_chunks_count: int
    documents: list[DocumentDebugItem]


def get_project_document_debug(db: Session, project_id: int) -> ProjectDocumentDebug:
    documents = list(
        db.execute(
            select(Document)
            .where(Document.project_id == project_id)
            .order_by(Document.created_at.desc())
        )
        .scalars()
        .all()
    )

    debug_documents: list[DocumentDebugItem] = []
    chunks_count = 0
    embedded_chunks_count = 0

    for document in documents:
        chunk_count, embedded_chunk_count = db.execute(
            select(
                func.count(DocumentChunk.id),
                func.count(DocumentChunk.embedding_json),
            )
            .where(DocumentChunk.project_id == project_id)
            .where(DocumentChunk.document_id == document.id)
        ).one()
        chunk_count = int(chunk_count or 0)
        embedded_chunk_count = int(embedded_chunk_count or 0)
        chunks_count += chunk_count
        embedded_chunks_count += embedded_chunk_count
        debug_documents.append(
            DocumentDebugItem(
                id=document.id,
                filename=document.filename,
                has_extracted_text=bool(document.extracted_text_path),
                chunk_count=chunk_count,
                embedded_chunk_count=embedded_chunk_count,
            )
        )

    return ProjectDocumentDebug(
        documents_count=len(documents),
        chunks_count=chunks_count,
        embedded_chunks_count=embedded_chunks_count,
        documents=debug_documents,
    )
