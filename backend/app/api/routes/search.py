from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
def semantic_search(
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    try:
        service = SearchService()

        results = service.search(
            db=db,
            workspace_id=request.workspace_id,
            query=request.query,
            limit=request.limit,
        )

        return SearchResponse(
            query=request.query,
            results=results,
            result_count=len(results),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

