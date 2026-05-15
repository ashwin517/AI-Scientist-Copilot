from dataclasses import dataclass

import requests

from app.config import settings


EMBEDDING_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: list[float] | None
    error: str | None = None


class OllamaEmbeddingClient:
    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        timeout_seconds: int = EMBEDDING_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model or settings.ollama_embedding_model
        self.url = url or settings.ollama_embedding_url
        self.timeout_seconds = timeout_seconds

    def embed_text(self, text: str) -> EmbeddingResult:
        if not text.strip():
            return EmbeddingResult(embedding=None, error="Cannot embed empty text.")

        try:
            response = requests.post(
                self.url,
                json={"model": self.model, "prompt": text},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return EmbeddingResult(embedding=None, error=str(exc))

        try:
            data = response.json()
        except ValueError:
            return EmbeddingResult(
                embedding=None,
                error="Ollama returned invalid JSON for embedding request.",
            )

        embedding = data.get("embedding")
        if not _is_embedding(embedding):
            return EmbeddingResult(
                embedding=None,
                error="Ollama embedding response did not include a numeric vector.",
            )

        return EmbeddingResult(embedding=[float(value) for value in embedding])


def _is_embedding(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (int, float)) for item in value
    )
