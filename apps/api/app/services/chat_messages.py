from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatMessage
from app.schemas.chat import ChatMessageCreate


def create_chat_message(
    db: Session,
    project_id: int,
    message_data: ChatMessageCreate,
) -> ChatMessage:
    message = ChatMessage(
        project_id=project_id,
        role=message_data.role,
        content=message_data.content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def list_project_chat_messages(
    db: Session,
    project_id: int,
) -> list[ChatMessage]:
    result = db.execute(
        select(ChatMessage)
        .where(ChatMessage.project_id == project_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    )
    return list(result.scalars().all())
