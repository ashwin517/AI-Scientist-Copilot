import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.ai.embedding_client import OllamaEmbeddingClient
from app.db.models import Document, DocumentChunk
from app.services.documents import ensure_document_text_extracted


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentProcessingResult:
    document_id: int
    project_id: int
    chunk_count: int
    embedded_chunk_count: int
    embedding_error_count: int
    embedding_errors: list[dict] = field(default_factory=list)


def process_document(
    db: Session,
    document: Document,
    embedding_client: OllamaEmbeddingClient | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> DocumentProcessingResult:
    text = read_extracted_document_text(db, document)
    if not text:
        delete_document_chunks(db, document)
        logger.info(
            "document processing no extracted text project_id=%s document_id=%s filename=%s",
            document.project_id,
            document.id,
            document.filename,
        )
        return DocumentProcessingResult(
            document_id=document.id,
            project_id=document.project_id,
            chunk_count=0,
            embedded_chunk_count=0,
            embedding_error_count=0,
        )

    chunks = split_text_into_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    client = embedding_client or OllamaEmbeddingClient()

    # If the document already has chunks, try to preserve them and only
    # fill in missing embeddings. If the existing chunks differ in count
    # or content, rebuild them.
    existing_chunks = list_document_chunks(db, document)

    embedding_errors_temp: list[tuple[object, str]] = []

    if existing_chunks and len(existing_chunks) == len(chunks):
        # Verify contents match; if not, fallback to rebuild.
        contents_match = all(
            existing_chunks[i].content == chunks[i] for i in range(len(chunks))
        )
        if contents_match:
            embedded_chunk_count = sum(
                1 for c in existing_chunks if c.embedding_json
            )
            embedding_error_count = 0

            for chunk_index, content in enumerate(chunks):
                existing = existing_chunks[chunk_index]
                if existing.embedding_json is None:
                    try:
                        embedding_result = client.embed_text(content)
                    except Exception as exc:  # pragma: no cover - defensive
                        embedding_result = None
                        err_msg = str(exc)

                    if embedding_result is None or embedding_result.embedding is None:
                        embedding_error_count += 1
                        err_msg = (
                            embedding_result.error if embedding_result is not None else err_msg
                        )
                        model_name = getattr(client, "model", None)
                        model_url = getattr(client, "url", None)
                        logger.warning(
                            "embedding failed project_id=%s document_id=%s chunk_index=%s model=%s url=%s error=%s",
                            document.project_id,
                            document.id,
                            chunk_index,
                            model_name,
                            model_url,
                            err_msg,
                        )
                        embedding_errors_temp.append((existing, err_msg or "unknown"))
                    else:
                        existing.embedding_json = json.dumps(
                            embedding_result.embedding
                        )
                        embedded_chunk_count += 1
                        db.add(existing)

            db.commit()
            # Build final embedding_errors list with ids for the preserved-chunks path
            embedding_errors: list[dict] = []
            for chunk_obj, err in embedding_errors_temp:
                chunk_id = getattr(chunk_obj, "id", None)
                embedding_errors.append({"chunk_id": chunk_id, "error": err})

            logger.info(
                "document processing complete project_id=%s document_id=%s filename=%s chunks=%s embedded_chunks=%s embedding_errors=%s",
                document.project_id,
                document.id,
                document.filename,
                len(chunks),
                embedded_chunk_count,
                embedding_error_count,
            )

            return DocumentProcessingResult(
                document_id=document.id,
                project_id=document.project_id,
                chunk_count=len(chunks),
                embedded_chunk_count=embedded_chunk_count,
                embedding_error_count=embedding_error_count,
                embedding_errors=embedding_errors,
            )
        else:
            # Contents changed: delete and rebuild below.
            delete_document_chunks(db, document, commit=False)
            embedded_chunk_count = 0
            embedding_error_count = 0

            for chunk_index, content in enumerate(chunks):
                try:
                    embedding_result = client.embed_text(content)
                except Exception as exc:  # pragma: no cover - defensive
                    embedding_result = None
                    err_msg = str(exc)

                embedding_json = None
                if embedding_result is None or embedding_result.embedding is None:
                    embedding_error_count += 1
                    err_msg = (
                        embedding_result.error if embedding_result is not None else err_msg
                    )
                    model_name = getattr(client, "model", None)
                    model_url = getattr(client, "url", None)
                    logger.warning(
                        "embedding failed project_id=%s document_id=%s chunk_index=%s model=%s url=%s error=%s",
                        document.project_id,
                        document.id,
                        chunk_index,
                        model_name,
                        model_url,
                        err_msg,
                    )
                    # Record against the chunk object; id will be available after commit
                    temp_chunk = DocumentChunk(
                        document_id=document.id,
                        project_id=document.project_id,
                        chunk_index=chunk_index,
                        content=content,
                        char_count=len(content),
                        embedding_json=None,
                    )
                    db.add(temp_chunk)
                    embedding_errors_temp.append((temp_chunk, err_msg or "unknown"))
                    continue
                else:
                    embedded_chunk_count += 1
                    embedding_json = json.dumps(embedding_result.embedding)

                db.add(
                    DocumentChunk(
                        document_id=document.id,
                        project_id=document.project_id,
                        chunk_index=chunk_index,
                        content=content,
                        char_count=len(content),
                        embedding_json=embedding_json,
                    )
                )

            db.commit()
            # Build final embedding_errors list with ids
            embedding_errors: list[dict] = []
            for chunk_obj, err in embedding_errors_temp:
                chunk_id = getattr(chunk_obj, "id", None)
                embedding_errors.append({"chunk_id": chunk_id, "error": err})

            return DocumentProcessingResult(
                document_id=document.id,
                project_id=document.project_id,
                chunk_count=len(chunks),
                embedded_chunk_count=embedded_chunk_count,
                embedding_error_count=embedding_error_count,
                embedding_errors=embedding_errors,
            )
    else:
        # No existing chunks or different count: (re)build all chunks.
        delete_document_chunks(db, document, commit=False)
        embedded_chunk_count = 0
        embedding_error_count = 0

        for chunk_index, content in enumerate(chunks):
            try:
                embedding_result = client.embed_text(content)
            except Exception as exc:  # pragma: no cover - defensive
                embedding_result = None
                err_msg = str(exc)

            embedding_json = None
            if embedding_result is None or embedding_result.embedding is None:
                embedding_error_count += 1
                err_msg = (
                    embedding_result.error if embedding_result is not None else err_msg
                )
                model_name = getattr(client, "model", None)
                model_url = getattr(client, "url", None)
                logger.warning(
                    "embedding failed project_id=%s document_id=%s chunk_index=%s model=%s url=%s error=%s",
                    document.project_id,
                    document.id,
                    chunk_index,
                    model_name,
                    model_url,
                    err_msg,
                )
                # Record the temp chunk so we can report id after commit
                temp_chunk = DocumentChunk(
                    document_id=document.id,
                    project_id=document.project_id,
                    chunk_index=chunk_index,
                    content=content,
                    char_count=len(content),
                    embedding_json=None,
                )
                db.add(temp_chunk)
                embedding_errors_temp.append((temp_chunk, err_msg or "unknown"))
                continue
            else:
                embedded_chunk_count += 1
                embedding_json = json.dumps(embedding_result.embedding)

            db.add(
                DocumentChunk(
                    document_id=document.id,
                    project_id=document.project_id,
                    chunk_index=chunk_index,
                    content=content,
                    char_count=len(content),
                    embedding_json=embedding_json,
                )
            )

        db.commit()
        # Build final embedding_errors list with ids
        embedding_errors: list[dict] = []
        for chunk_obj, err in embedding_errors_temp:
            chunk_id = getattr(chunk_obj, "id", None)
            embedding_errors.append({"chunk_id": chunk_id, "error": err})

        return DocumentProcessingResult(
            document_id=document.id,
            project_id=document.project_id,
            chunk_count=len(chunks),
            embedded_chunk_count=embedded_chunk_count,
            embedding_error_count=embedding_error_count,
            embedding_errors=embedding_errors,
        )
    logger.info(
        "document processing complete project_id=%s document_id=%s filename=%s chunks=%s embedded_chunks=%s embedding_errors=%s",
        document.project_id,
        document.id,
        document.filename,
        len(chunks),
        embedded_chunk_count,
        embedding_error_count,
    )
    # Fallback return (shouldn't normally be reached because branches return).
    embedding_errors: list[dict] = []
    for chunk_obj, err in embedding_errors_temp:
        chunk_id = getattr(chunk_obj, "id", None)
        embedding_errors.append({"chunk_id": chunk_id, "error": err})

    return DocumentProcessingResult(
        document_id=document.id,
        project_id=document.project_id,
        chunk_count=len(chunks),
        embedded_chunk_count=embedded_chunk_count,
        embedding_error_count=embedding_error_count,
        embedding_errors=embedding_errors,
    )


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be zero or less than chunk_size.")

    chunks: list[str] = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = normalized_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = end - chunk_overlap

    return chunks


def list_document_chunks(db: Session, document: Document) -> list[DocumentChunk]:
    result = db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
    )
    return list(result.scalars().all())


def delete_document_chunks(
    db: Session,
    document: Document,
    commit: bool = True,
) -> None:
    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    if commit:
        db.commit()


def read_extracted_document_text(db: Session, document: Document) -> str:
    extracted_text_path = ensure_document_text_extracted(db, document)
    if not extracted_text_path:
        return ""

    text_path = Path(extracted_text_path)
    if not text_path.exists():
        return ""

    return text_path.read_text(encoding="utf-8", errors="ignore")
