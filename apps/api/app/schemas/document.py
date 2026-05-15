from datetime import datetime

from pydantic import BaseModel, Field


class DocumentRead(BaseModel):
    id: int
    project_id: int
    filename: str
    file_path: str
    mime_type: str
    file_size: int
    extracted_text_path: str | None
    created_at: datetime


class DocumentProcessingRead(BaseModel):
    document_id: int
    project_id: int
    chunk_count: int
    embedded_chunk_count: int
    embedding_error_count: int


class DocumentChunkPreviewRead(BaseModel):
    id: int
    chunk_index: int
    content_preview: str
    char_count: int
    has_embedding: bool


class DocumentChunksRead(BaseModel):
    document_id: int
    project_id: int
    total_chunks: int
    chunks: list[DocumentChunkPreviewRead]


class DocumentDebugItemRead(BaseModel):
    id: int
    filename: str
    has_extracted_text: bool
    chunk_count: int
    embedded_chunk_count: int


class ProjectDocumentDebugRead(BaseModel):
    documents_count: int
    chunks_count: int
    embedded_chunks_count: int
    documents: list[DocumentDebugItemRead]


class DocumentSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class DocumentSearchResult(BaseModel):
    document_id: int
    filename: str
    chunk_id: int
    chunk_index: int
    score: float
    content_preview: str


class DocumentSearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[DocumentSearchResult]


class DocumentAskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class DocumentAskResponse(BaseModel):
    answer: str
    sources: list[DocumentSearchResult]


class DocumentUploadResult(BaseModel):
    document: DocumentRead
    processing: DocumentProcessingRead | None = None
