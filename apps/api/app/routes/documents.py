from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.ai.rag_service import answer_document_question
from app.ai.retrieval_service import retrieve_relevant_chunks
from app.db.session import get_db
from app.schemas.document import (
    DocumentAskRequest,
    DocumentAskResponse,
    DocumentChunkPreviewRead,
    DocumentChunksRead,
    DocumentDebugItemRead,
    DocumentProcessingRead,
    DocumentRead,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentSearchResult,
    DocumentUploadResult,
    ProjectDocumentDebugRead,
)
from app.services.document_chunks import list_document_chunks, process_document
from app.services.document_debug import get_project_document_debug
from app.services.documents import (
    delete_document,
    document_to_read_data,
    get_document,
    list_project_documents,
    save_uploaded_document,
)
from app.services.projects import get_project


router = APIRouter(tags=["documents"])


@router.post(
    "/projects/{project_id}/documents/upload",
    response_model=DocumentUploadResult,
    status_code=status.HTTP_201_CREATED,
)
def upload_project_document_route(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentUploadResult:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    try:
        document = save_uploaded_document(db, project_id, file)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    processing = process_document(db, document)

    return DocumentUploadResult(
        document=document_to_read_data(document),
        processing=DocumentProcessingRead(**processing.__dict__),
    )


@router.get(
    "/projects/{project_id}/documents",
    response_model=list[DocumentRead],
)
def list_project_documents_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> list[DocumentRead]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return [
        document_to_read_data(document)
        for document in list_project_documents(db, project_id)
    ]


@router.post(
    "/projects/{project_id}/documents/search",
    response_model=DocumentSearchResponse,
)
def search_project_documents_route(
    project_id: int,
    search_request: DocumentSearchRequest,
    db: Session = Depends(get_db),
) -> DocumentSearchResponse:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    results = retrieve_relevant_chunks(
        db,
        project_id=project_id,
        query=search_request.query,
        top_k=search_request.top_k,
    )
    return DocumentSearchResponse(
        query=search_request.query,
        top_k=search_request.top_k,
        results=[
            DocumentSearchResult(
                document_id=result.document_id,
                filename=result.filename,
                chunk_id=result.chunk_id,
                chunk_index=result.chunk_index,
                score=result.score,
                content_preview=result.content_preview,
            )
            for result in results
        ],
    )


@router.post(
    "/projects/{project_id}/documents/ask",
    response_model=DocumentAskResponse,
)
def ask_project_documents_route(
    project_id: int,
    ask_request: DocumentAskRequest,
    db: Session = Depends(get_db),
) -> DocumentAskResponse:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    answer = answer_document_question(
        db,
        project_id=project_id,
        question=ask_request.question,
        top_k=ask_request.top_k,
    )
    return DocumentAskResponse(
        answer=answer.answer,
        sources=[
            DocumentSearchResult(
                document_id=source.document_id,
                filename=source.filename,
                chunk_id=source.chunk_id,
                chunk_index=source.chunk_index,
                score=source.score,
                content_preview=source.content_preview,
            )
            for source in answer.sources
        ],
    )


@router.get(
    "/projects/{project_id}/documents/debug",
    response_model=ProjectDocumentDebugRead,
)
def get_project_documents_debug_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> ProjectDocumentDebugRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    debug = get_project_document_debug(db, project_id)
    return ProjectDocumentDebugRead(
        documents_count=debug.documents_count,
        chunks_count=debug.chunks_count,
        embedded_chunks_count=debug.embedded_chunks_count,
        documents=[
            DocumentDebugItemRead(
                id=document.id,
                filename=document.filename,
                has_extracted_text=document.has_extracted_text,
                chunk_count=document.chunk_count,
                embedded_chunk_count=document.embedded_chunk_count,
            )
            for document in debug.documents
        ],
    )


@router.get(
    "/projects/{project_id}/documents/{document_id}",
    response_model=DocumentRead,
)
def get_project_document_route(
    project_id: int,
    document_id: int,
    db: Session = Depends(get_db),
) -> DocumentRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    document = get_document(db, document_id)
    if document is None or document.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document_to_read_data(document)


@router.get(
    "/projects/{project_id}/documents/{document_id}/chunks",
    response_model=DocumentChunksRead,
)
def list_project_document_chunks_route(
    project_id: int,
    document_id: int,
    db: Session = Depends(get_db),
) -> DocumentChunksRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    document = get_document(db, document_id)
    if document is None or document.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    chunks = list_document_chunks(db, document)
    return DocumentChunksRead(
        document_id=document.id,
        project_id=document.project_id,
        total_chunks=len(chunks),
        chunks=[
            DocumentChunkPreviewRead(
                id=chunk.id,
                chunk_index=chunk.chunk_index,
                content_preview=chunk.content[:300],
                char_count=chunk.char_count,
                has_embedding=chunk.embedding_json is not None,
            )
            for chunk in chunks
        ],
    )


@router.post(
    "/projects/{project_id}/documents/{document_id}/process",
    response_model=DocumentProcessingRead,
)
def process_project_document_route(
    project_id: int,
    document_id: int,
    db: Session = Depends(get_db),
) -> DocumentProcessingRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    document = get_document(db, document_id)
    if document is None or document.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    processing = process_document(db, document)
    return DocumentProcessingRead(**processing.__dict__)


@router.delete(
    "/projects/{project_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_project_document_route(
    project_id: int,
    document_id: int,
    db: Session = Depends(get_db),
) -> None:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    document = get_document(db, document_id)
    if document is None or document.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    delete_document(db, document)
