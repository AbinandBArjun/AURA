from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    workspace_id: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    document_id: int
    filename: str
    chunk_id: int
    page_start: int
    page_end: int
    score: float


class RAGResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    evidence_count: int
