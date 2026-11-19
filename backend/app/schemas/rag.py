from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RAGRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    source_id: str
    document_id: int
    filename: str
    chunk_id: int
    page_start: int
    page_end: int
    score: float


class ClaimVerification(BaseModel):
    claim_id: str
    claim_text: str
    status: Literal[
        "supported",
        "partially_supported",
        "contradicted",
        "insufficient_evidence",
    ]
    rationale: str
    evidence_ids: list[str]


class RAGResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    evidence_count: int
    claims: list[ClaimVerification] = Field(default_factory=list)
