from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.rag import RAGRequest, RAGResponse
from app.services.rag_service import RAGService, RAGError
from app.services.search_service import SearchService


router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/answer", response_model=RAGResponse)
def generate_answer(
    request: RAGRequest,
    db: Session = Depends(get_db),
):
    try:
        service = RAGService(
            search_service=SearchService(),
        )

        result = service.answer(
            workspace_id=request.workspace_id,
            question=request.question,
            limit=request.limit,
        )

        return result

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except RAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
