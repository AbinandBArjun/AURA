import pytest

from app.models import ResearchRun
from app.schemas.research import ResearchResponse
from app.services.research_service import (
    ResearchExecutionError,
    ResearchService,
)


class FakeResearchGraph:
    def __init__(self, state=None, error=None):
        self.state = state
        self.error = error
        self.received_input = None

    def invoke(self, state):
        self.received_input = state

        if self.error:
            raise self.error

        return self.state


def build_successful_state():
    evidence = [
        {
            "citation_id": "S1",
            "document_id": 1,
            "filename": "research.pdf",
            "chunk_id": 10,
            "page_start": 1,
            "page_end": 2,
            "score": 0.95,
            "text": "Evidence supporting claim one.",
        },
        {
            "citation_id": "S2",
            "document_id": 1,
            "filename": "research.pdf",
            "chunk_id": 11,
            "page_start": 3,
            "page_end": 4,
            "score": 0.82,
            "text": "Unreferenced evidence.",
        },
    ]

    claims = [
        {
            "claim_id": "C1",
            "claim_text": "Research claim one.",
            "evidence_ids": ["S1"],
        }
    ]

    verification_results = [
        {
            "claim_id": "C1",
            "status": "supported",
            "rationale": "The evidence supports this claim.",
            "evidence_ids": ["S1"],
        }
    ]

    return {
        "workspace_id": 1,
        "question": "Research question",
        "research_plan": ["Find relevant evidence"],
        "evidence": evidence,
        "findings": "Research findings.",
        "claims": claims,
        "verification_results": verification_results,
        "report": "# Research Report",
    }


def test_research_service_completes_successfully(
    db_session,
    test_workspace,
):
    graph = FakeResearchGraph(state=build_successful_state())

    service = ResearchService(
        db=db_session,
        graph=graph,
    )

    result = service.run(
        workspace_id=test_workspace.id,
        question="Research question",
    )

    assert isinstance(result, ResearchResponse)
    assert result.status == "completed"
    assert result.question == "Research question"
    assert result.findings == "Research findings."
    assert result.report == "# Research Report"
    assert result.evidence_count == 2

    assert graph.received_input == {
        "workspace_id": test_workspace.id,
        "question": "Research question",
    }

    run = db_session.get(ResearchRun, result.research_run_id)

    assert run is not None
    assert run.status == "completed"
    assert run.result_summary == "Research findings."
    assert run.completed_at is not None


def test_research_service_returns_only_referenced_sources(
    db_session,
    test_workspace,
):
    graph = FakeResearchGraph(state=build_successful_state())

    service = ResearchService(
        db=db_session,
        graph=graph,
    )

    result = service.run(
        workspace_id=test_workspace.id,
        question="Research question",
    )

    assert len(result.sources) == 1
    assert result.sources[0].citation_id == "S1"
    assert result.sources[0].document_id == 1

    # Raw evidence text must not be exposed in source metadata.
    assert not hasattr(result.sources[0], "text")


def test_research_service_rejects_missing_workspace(db_session):
    graph = FakeResearchGraph(state=build_successful_state())

    service = ResearchService(
        db=db_session,
        graph=graph,
    )

    with pytest.raises(ValueError, match="Workspace not found"):
        service.run(
            workspace_id=999,
            question="Research question",
        )

    assert db_session.query(ResearchRun).count() == 0
    assert graph.received_input is None


def test_research_service_marks_run_failed_when_graph_raises(
    db_session,
    test_workspace,
):
    graph = FakeResearchGraph(
        error=RuntimeError("Internal model error")
    )

    service = ResearchService(
        db=db_session,
        graph=graph,
    )

    with pytest.raises(
        ResearchExecutionError,
        match="Research workflow execution failed",
    ):
        service.run(
            workspace_id=test_workspace.id,
            question="Research question",
        )

    run = (
        db_session.query(ResearchRun)
        .filter_by(workspace_id=test_workspace.id)
        .one()
    )

    assert run.status == "failed"
    assert run.error_message == "Research workflow execution failed."
    assert run.completed_at is not None


def test_research_service_marks_run_failed_for_invalid_graph_output(
    db_session,
    test_workspace,
):
    graph = FakeResearchGraph(state=None)

    service = ResearchService(
        db=db_session,
        graph=graph,
    )

    with pytest.raises(ResearchExecutionError):
        service.run(
            workspace_id=test_workspace.id,
            question="Research question",
        )

    run = (
        db_session.query(ResearchRun)
        .filter_by(workspace_id=test_workspace.id)
        .one()
    )

    assert run.status == "failed"
    assert run.completed_at is not None
