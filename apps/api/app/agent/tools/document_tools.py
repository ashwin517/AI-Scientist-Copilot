import logging
from typing import Any

from sqlalchemy.orm import Session

from app.ai.rag_service import answer_document_question as generate_document_answer
from app.services.document_debug import get_project_document_debug


NO_DOCUMENTS_MESSAGE = (
    "This project has no uploaded documents yet. Please upload a PDF or document first."
)
DOCUMENT_NOT_PROCESSED_MESSAGE = (
    "Your document was uploaded but has not been processed yet."
)
EMBEDDINGS_MISSING_MESSAGE = (
    "Your document text was extracted, but embeddings were not generated yet."
)
NO_RELEVANT_CHUNKS_MESSAGE = (
    "I could not retrieve relevant sections from the uploaded documents."
)
logger = logging.getLogger(__name__)


def answer_document_question(
    db: Session,
    project_id: int,
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:
    debug = get_project_document_debug(db, project_id)
    logger.info(
        "agent document rag state project_id=%s documents=%s chunks=%s embedded_chunks=%s question_length=%s",
        project_id,
        debug.documents_count,
        debug.chunks_count,
        debug.embedded_chunks_count,
        len(question),
    )

    if debug.documents_count == 0:
        raise ValueError(NO_DOCUMENTS_MESSAGE)
    if debug.chunks_count == 0:
        return {
            "answer": DOCUMENT_NOT_PROCESSED_MESSAGE,
            "sources": [],
        }
    if debug.embedded_chunks_count == 0:
        return {
            "answer": EMBEDDINGS_MISSING_MESSAGE,
            "sources": [],
        }

    rag_answer = generate_document_answer(
        db,
        project_id=project_id,
        question=question,
        top_k=top_k,
    )
    sources = [
        {
            "document_id": source.document_id,
            "filename": source.filename,
            "chunk_id": source.chunk_id,
            "chunk_index": source.chunk_index,
            "score": source.score,
            "content_preview": source.content_preview,
        }
        for source in rag_answer.sources
    ]
    logger.info(
        "agent document rag result project_id=%s retrieval_results=%s source_filenames=%s",
        project_id,
        len(sources),
        [source["filename"] for source in sources],
    )

    if not sources:
        return {
            "answer": NO_RELEVANT_CHUNKS_MESSAGE,
            "sources": [],
        }

    return {
        "answer": rag_answer.answer,
        "sources": sources,
    }
