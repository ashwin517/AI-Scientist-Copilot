import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/ai_scientist",
    )
    storage_dir: str = os.getenv("STORAGE_DIR", "storage")
    ollama_embedding_url: str = os.getenv(
        "OLLAMA_EMBEDDING_URL",
        "http://localhost:11434/api/embeddings",
    )
    ollama_embedding_model: str = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text",
    )
    ollama_chat_url: str = os.getenv(
        "OLLAMA_CHAT_URL",
        "http://localhost:11434/api/generate",
    )
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")


settings = Settings()
