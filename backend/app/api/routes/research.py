from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from app.agents.graph import build_research_graph
from app.db.session import get_db
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.research_service import (
    ResearchExecutionError,
    ResearchService,
)


router = APIRouter(prefix="/research", tags=["Research"])


@lru_cache(maxsize=1)
def get_research_graph() -> CompiledStateGraph:
    """Return the cached research graph."""
    return build_research_graph()


@router.post("/run", response_model=ResearchResponse)
def run_research(
    request: ResearchRequest,
    db: Session = Depends(get_db),
    graph: CompiledStateGraph = Depends(get_research_graph),
):
    service = ResearchService(db=db, graph=graph)

    try:
        return service.run(
            workspace_id=request.workspace_id,
            question=request.question,
        )

    except ValueError as exc:
        status_code = 404 if str(exc) == "Workspace not found." else 422
        raise HTTPException(
            status_code=status_code,
            detail=str(exc),
        ) from exc

    except ResearchExecutionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Research workflow execution failed.",
        ) from exc
