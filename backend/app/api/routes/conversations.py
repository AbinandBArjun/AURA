from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_workspace
from app.db.session import get_db
from app.models import Workspace
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationMessageCreateRequest,
    ConversationResponse,
    ConversationTurnResponse,
)
from app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)
from app.services.rag_service import RAGError


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
    workspace: Workspace = Depends(get_current_workspace),
):
    service = ConversationService(
        db=db,
        workspace_id=workspace.id,
    )

    return service.create_conversation(title=request.title)


@router.get(
    "",
    response_model=list[ConversationResponse],
)
def list_conversations(
    db: Session = Depends(get_db),
    workspace: Workspace = Depends(get_current_workspace),
):
    service = ConversationService(
        db=db,
        workspace_id=workspace.id,
    )

    return service.list_conversations()


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    workspace: Workspace = Depends(get_current_workspace),
):
    service = ConversationService(
        db=db,
        workspace_id=workspace.id,
    )

    try:
        return service.get_conversation(conversation_id)

    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationTurnResponse,
    status_code=status.HTTP_201_CREATED,
)
def send_message(
    conversation_id: int,
    request: ConversationMessageCreateRequest,
    db: Session = Depends(get_db),
    workspace: Workspace = Depends(get_current_workspace),
):
    service = ConversationService(
        db=db,
        workspace_id=workspace.id,
    )

    try:
        return service.send_message(
            conversation_id=conversation_id,
            content=request.content,
            limit=request.limit,
        )

    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    except RAGError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
