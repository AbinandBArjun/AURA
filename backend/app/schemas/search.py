from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    workspace_id: int = Field(gt=0)
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    document_id: int
    filename: str
    chunk_id: int
    text: str
    page_start: int
    page_end: int
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    result_count: int

