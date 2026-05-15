from dataclasses import dataclass
import logging

import requests
from sqlalchemy.orm import Session

from app.ai.retrieval_service import RetrievedChunk, retrieve_relevant_chunks
from app.config import settings


REQUEST_TIMEOUT_SECONDS = 60
MAX_QUESTION_CHARACTERS = 2000
MAX_CHUNK_CHARACTERS = 1200
MAX_PROMPT_CHARACTERS = 14000

NO_RELEVANT_CHUNKS_ANSWER = (
    "I could not find relevant document chunks for this question in the "
    "project. I cannot answer from the uploaded documents yet."
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagSource:
    document_id: int
    filename: str
    chunk_id: int
    chunk_index: int
    score: float
    content_preview: str


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RagSource]


def answer_document_question(
    db: Session,
    project_id: int,
    question: str,
    top_k: int = 5,
) -> RagAnswer:
    chunks = retrieve_relevant_chunks(
        db,
        project_id=project_id,
        query=question,
        top_k=top_k,
    )
    sources = [_chunk_to_source(chunk) for chunk in chunks]
    logger.info(
        "rag document answer retrieval project_id=%s top_k=%s question_length=%s result_count=%s filenames=%s",
        project_id,
        top_k,
        len(question),
        len(chunks),
        [source.filename for source in sources],
    )
    if not chunks:
        return RagAnswer(answer=NO_RELEVANT_CHUNKS_ANSWER, sources=[])

    prompt = build_grounded_prompt(question, chunks)
    try:
        response = requests.post(
            settings.ollama_chat_url,
            json={
                "model": settings.ollama_chat_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return RagAnswer(
            answer=(
                "I found relevant document chunks, but I could not reach the "
                "local Ollama chat model to generate a grounded answer."
            ),
            sources=sources,
        )

    try:
        data = response.json()
    except ValueError:
        return RagAnswer(
            answer=(
                "I found relevant document chunks, but Ollama returned an "
                "invalid response while generating the answer."
            ),
            sources=sources,
        )

    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        return RagAnswer(
            answer=(
                "I found relevant document chunks, but Ollama did not return a "
                "usable answer."
            ),
            sources=sources,
        )

    return RagAnswer(answer=answer.strip(), sources=sources)


def build_grounded_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    chunk_context = "\n\n".join(
        _format_chunk_for_prompt(chunk)
        for chunk in chunks
    )
    prompt = f"""System instruction:
You are AI Scientist Copilot answering a question from retrieved document chunks.

Grounding rules:
- Answer only from the provided chunks.
- If the chunks do not contain enough information, say so plainly.
- Do not invent facts, sources, or citations.
- Cite sources using exactly this format: [doc:filename chunk:X].
- Use only chunk citations that appear in the provided chunk list.
- Keep the answer concise.

Retrieved chunks:
{chunk_context}

Question:
{_truncate_text(question.strip(), MAX_QUESTION_CHARACTERS)}

Grounded answer:"""

    return _truncate_text(prompt, MAX_PROMPT_CHARACTERS)


def _format_chunk_for_prompt(chunk: RetrievedChunk) -> str:
    citation = f"[doc:{chunk.filename} chunk:{chunk.chunk_index}]"
    content = _truncate_text(chunk.content, MAX_CHUNK_CHARACTERS)
    return (
        f"Source: {citation}\n"
        f"Document ID: {chunk.document_id}\n"
        f"Chunk ID: {chunk.chunk_id}\n"
        f"Similarity score: {chunk.score:.4f}\n"
        f"Content:\n{content}"
    )


def _chunk_to_source(chunk: RetrievedChunk) -> RagSource:
    return RagSource(
        document_id=chunk.document_id,
        filename=chunk.filename,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        score=chunk.score,
        content_preview=chunk.content_preview,
    )


def _truncate_text(value: str, max_characters: int) -> str:
    if len(value) <= max_characters:
        return value
    omitted = len(value) - max_characters
    return value[:max_characters].rstrip() + f"\n...[truncated {omitted} characters]"
