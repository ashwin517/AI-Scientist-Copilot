from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import (
    ChatMessageCreate,
    ChatMessageRead,
    ChatRequest,
    ChatResponse,
)
from app.services.chat_messages import (
    create_chat_message,
    list_project_chat_messages,
)
from app.services.llm_chat import generate_chat_reply
from app.services.projects import get_project


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat_route(chat_request: ChatRequest) -> ChatResponse:
    return generate_chat_reply(
        message=chat_request.message,
        dataset_summary=chat_request.dataset_summary,
        model_result=chat_request.model_result,
    )


@router.get("/projects/{project_id}/chat", response_model=list[ChatMessageRead])
def list_project_chat_route(
    project_id: int,
    db: Session = Depends(get_db),
) -> list[ChatMessageRead]:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return list_project_chat_messages(db, project_id)


@router.post(
    "/projects/{project_id}/chat",
    response_model=ChatMessageRead,
    status_code=status.HTTP_201_CREATED,
)
def create_project_chat_message_route(
    project_id: int,
    message_data: ChatMessageCreate,
    db: Session = Depends(get_db),
) -> ChatMessageRead:
    if get_project(db, project_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    return create_chat_message(db, project_id, message_data)
