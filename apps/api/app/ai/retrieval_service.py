import json
import logging
import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.embedding_client import OllamaEmbeddingClient
from app.db.models import Document, DocumentChunk


MAX_TOP_K = 20
CONTENT_PREVIEW_CHARACTERS = 500
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: int
    filename: str
    chunk_id: int
    chunk_index: int
    score: float
    content: str
    content_preview: str


def retrieve_relevant_chunks(
    db: Session,
    project_id: int,
    query: str,
    top_k: int = 5,
    embedding_client: OllamaEmbeddingClient | None = None,
) -> list[RetrievedChunk]:
    query = query.strip()
    if not query:
        logger.info(
            "document retrieval skipped empty query project_id=%s top_k=%s",
            project_id,
            top_k,
        )
        return []

    top_k = max(1, min(top_k, MAX_TOP_K))
    client = embedding_client or OllamaEmbeddingClient()
    query_embedding_result = client.embed_text(query)
    if query_embedding_result.embedding is None:
        logger.info(
            "document retrieval query embedding failed project_id=%s top_k=%s error_present=%s",
            project_id,
            top_k,
            bool(query_embedding_result.error),
        )
        return []

    rows = db.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.project_id == project_id)
        .where(Document.project_id == project_id)
        .where(DocumentChunk.embedding_json.is_not(None))
    ).all()
    logger.info(
        "document retrieval candidate chunks project_id=%s top_k=%s embedded_candidates=%s",
        project_id,
        top_k,
        len(rows),
    )

    scored_chunks: list[RetrievedChunk] = []
    for chunk, filename in rows:
        chunk_embedding = _decode_embedding(chunk.embedding_json)
        if chunk_embedding is None:
            continue

        score = cosine_similarity(query_embedding_result.embedding, chunk_embedding)
        scored_chunks.append(
            RetrievedChunk(
                document_id=chunk.document_id,
                filename=filename,
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                score=score,
                content=chunk.content,
                content_preview=chunk.content[:CONTENT_PREVIEW_CHARACTERS],
            )
        )

    results = sorted(scored_chunks, key=lambda result: result.score, reverse=True)[:top_k]
    logger.info(
        "document retrieval results project_id=%s result_count=%s scores=%s filenames=%s",
        project_id,
        len(results),
        [round(result.score, 4) for result in results],
        [result.filename for result in results],
    )
    return results


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _decode_embedding(value: str | None) -> list[float] | None:
    if value is None:
        return None

    try:
        embedding = json.loads(value)
    except json.JSONDecodeError:
        return None

    if not isinstance(embedding, list) or not all(
        isinstance(item, (int, float)) for item in embedding
    ):
        return None

    return [float(item) for item in embedding]
