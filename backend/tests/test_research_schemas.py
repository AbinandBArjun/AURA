import pytest
from pydantic import ValidationError

from app.schemas.research import (
    ResearchRequest,
    ResearchResponse,
)


def test_research_request_accepts_valid_data():
    request = ResearchRequest(
        workspace_id=1,
        question="What are the latest advances in RAG?",
    )

    assert request.workspace_id == 1
    assert request.question == "What are the latest advances in RAG?"


@pytest.mark.parametrize("workspace_id", [0, -1])
def test_research_request_rejects_invalid_workspace_id(workspace_id):
    with pytest.raises(ValidationError):
        ResearchRequest(
            workspace_id=workspace_id,
            question="Research question",
        )


def test_research_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        ResearchRequest(
            workspace_id=1,
            question="",
        )


def test_research_request_rejects_question_over_2000_characters():
    with pytest.raises(ValidationError):
        ResearchRequest(
            workspace_id=1,
            question="a" * 2001,
        )


def test_research_response_accepts_empty_evidence_results():
    response = ResearchResponse(
        research_run_id=1,
        status="completed",
        question="Research question",
        research_plan=["Find relevant evidence"],
        findings="Insufficient evidence was retrieved.",
        report="# Research Report",
        claims=[],
        verification_results=[],
        sources=[],
        evidence_count=0,
    )

    assert response.status == "completed"
    assert response.claims == []
    assert response.verification_results == []
    assert response.sources == []
    assert response.evidence_count == 0
