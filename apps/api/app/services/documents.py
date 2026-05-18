import logging
import re
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Document
from app.schemas.document import DocumentRead
from app.services.id_reset import reset_empty_id_sequences
from app.services.memory_service import delete_memory, update_project_summary, upsert_memory


ALLOWED_DOCUMENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/x-markdown",
}
TEXT_MIME_TYPES = {"text/plain", "text/markdown", "application/x-markdown"}
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
logger = logging.getLogger(__name__)


def save_uploaded_document(
    db: Session,
    project_id: int,
    file: UploadFile,
) -> Document:
    filename = _safe_filename(file.filename or "uploaded-document")
    mime_type = file.content_type or "application/octet-stream"
    suffix = Path(filename).suffix.lower()

    if mime_type not in ALLOWED_DOCUMENT_TYPES and suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported document type. Upload a PDF, TXT, or Markdown file.")

    project_dir = Path(settings.storage_dir) / "projects" / str(project_id) / "documents"
    project_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid4().hex}{suffix or '.bin'}"
    file_path = project_dir / stored_filename
    with file_path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)

    file_size = file_path.stat().st_size
    extracted_text_path = _extract_text_if_possible(file_path, mime_type)

    document = Document(
        project_id=project_id,
        filename=filename,
        file_path=str(file_path),
        mime_type=mime_type,
        file_size=file_size,
        extracted_text_path=str(extracted_text_path) if extracted_text_path else None,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    document_count = _project_document_count(db, project_id)
    upsert_memory(
        db,
        project_id,
        "latest_document_id",
        document.id,
        memory_type="document",
        source="document_upload",
    )
    upsert_memory(
        db,
        project_id,
        "latest_document_filename",
        document.filename,
        memory_type="document",
        source="document_upload",
    )
    upsert_memory(
        db,
        project_id,
        "document_count",
        document_count,
        memory_type="document",
        source="document_upload",
    )
    update_project_summary(db, project_id)
    logger.info(
        "document uploaded project_id=%s document_id=%s filename=%s mime_type=%s file_size=%s has_extracted_text=%s",
        project_id,
        document.id,
        document.filename,
        document.mime_type,
        document.file_size,
        document.extracted_text_path is not None,
    )
    return document


def list_project_documents(db: Session, project_id: int) -> list[Document]:
    result = db.execute(
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


def get_document(db: Session, document_id: int) -> Document | None:
    return db.get(Document, document_id)


def delete_document(db: Session, document: Document) -> None:
    file_paths = [
        Path(document.file_path),
        Path(document.extracted_text_path) if document.extracted_text_path else None,
    ]
    db.delete(document)
    db.commit()
    _sync_document_memory_after_delete(db, document.project_id)
    reset_empty_id_sequences(db, [Document.__tablename__])
    db.commit()

    for file_path in file_paths:
        if file_path is None:
            continue
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass


def _sync_document_memory_after_delete(db: Session, project_id: int) -> None:
    documents = list_project_documents(db, project_id)
    document_count = len(documents)
    upsert_memory(
        db,
        project_id,
        "document_count",
        document_count,
        memory_type="document",
        source="document_delete",
    )
    if documents:
        latest_document = documents[0]
        upsert_memory(
            db,
            project_id,
            "latest_document_id",
            latest_document.id,
            memory_type="document",
            source="document_delete",
        )
        upsert_memory(
            db,
            project_id,
            "latest_document_filename",
            latest_document.filename,
            memory_type="document",
            source="document_delete",
        )
        return

    delete_memory(db, project_id, "latest_document_id")
    delete_memory(db, project_id, "latest_document_filename")
    update_project_summary(db, project_id)


def ensure_document_text_extracted(db: Session, document: Document) -> str | None:
    if document.extracted_text_path:
        extracted_path = Path(document.extracted_text_path)
        if extracted_path.exists():
            return document.extracted_text_path

    extracted_text_path = _extract_text_if_possible(
        Path(document.file_path),
        document.mime_type,
    )
    if extracted_text_path is None:
        return None

    document.extracted_text_path = str(extracted_text_path)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document.extracted_text_path


def document_to_read_data(document: Document) -> DocumentRead:
    return DocumentRead(
        id=document.id,
        project_id=document.project_id,
        filename=document.filename,
        file_path=document.file_path,
        mime_type=document.mime_type,
        file_size=document.file_size,
        extracted_text_path=document.extracted_text_path,
        created_at=document.created_at,
    )


def _extract_text_if_possible(file_path: Path, mime_type: str) -> Path | None:
    text: str | None = None
    suffix = file_path.suffix.lower()

    if mime_type in TEXT_MIME_TYPES or suffix in {".txt", ".md", ".markdown"}:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    elif mime_type == "application/pdf" or suffix == ".pdf":
        text = _extract_pdf_text(file_path)

    if not text:
        return None

    extracted_path = file_path.with_suffix(file_path.suffix + ".txt")
    extracted_path.write_text(text, encoding="utf-8")
    return extracted_path


def _extract_pdf_text(file_path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return None

    try:
        reader = PdfReader(str(file_path))
        page_text = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return None

    text = "\n\n".join(text.strip() for text in page_text if text.strip())
    return text or None


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "uploaded-document"
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    return name[:255] or "uploaded-document"


def _project_document_count(db: Session, project_id: int) -> int:
    result = db.execute(
        select(func.count()).select_from(Document).where(Document.project_id == project_id)
    )
    return int(result.scalar_one())
