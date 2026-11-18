from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    workspace_id: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=2000)


class ResearchClaim(BaseModel):
    claim_id: str
    claim_text: str
    evidence_ids: list[str]


class ResearchVerification(BaseModel):
    claim_id: str
    status: Literal[
        "supported",
        "partially_supported",
        "contradicted",
        "insufficient_evidence",
    ]
    rationale: str
    evidence_ids: list[str]


class ResearchSource(BaseModel):
    citation_id: str
    document_id: int
    filename: str
    chunk_id: int
    page_start: int | None = None
    page_end: int | None = None
    score: float


class ResearchResponse(BaseModel):
    research_run_id: int
    status: Literal["completed"]
    question: str
    research_plan: list[str]
    findings: str
    report: str
    claims: list[ResearchClaim]
    verification_results: list[ResearchVerification]
    sources: list[ResearchSource]
    evidence_count: int
