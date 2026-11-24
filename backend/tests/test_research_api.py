from contextlib import contextmanager

from app.api.routes.research import get_research_graph
from app.db.session import get_db
from app.main import app
from app.models import ResearchRun


class FakeResearchGraph:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def invoke(self, state):
        if self.should_fail:
            raise RuntimeError("Simulated graph failure")

        return {
            **state,
            "research_plan": ["Review available evidence"],
            "evidence": [
                {
                    "citation_id": "S1",
                    "document_id": 1,
                    "filename": "research.pdf",
                    "chunk_id": 1,
                    "page_start": 1,
                    "page_end": 2,
                    "score": 0.95,
                    "text": "Evidence supporting the research claim.",
                }
            ],
            "findings": "The evidence supports the research claim.",
            "claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "The evidence supports the research claim.",
                    "evidence_ids": ["S1"],
                }
            ],
            "verification_results": [
                {
                    "claim_id": "C1",
                    "status": "supported",
                    "rationale": "The evidence directly supports the claim.",
                    "evidence_ids": ["S1"],
                }
            ],
            "report": "# Research Report\n\nThe evidence supports the claim.",
        }


@contextmanager
def get_api_db_session():
    """Access the same test database used by the API client."""
    db_generator = app.dependency_overrides[get_db]()
    db = next(db_generator)

    try:
        yield db
    finally:
        db_generator.close()


def test_research_api_returns_completed_report(client, test_workspace):
    app.dependency_overrides[get_research_graph] = (
        lambda: FakeResearchGraph()
    )

    question = "What does the research evidence show?"

    response = client.post(
        "/api/v1/research/run",
        json={
            "workspace_id": test_workspace.id,
            "question": question,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["question"] == question
    assert data["research_plan"] == ["Review available evidence"]
    assert len(data["claims"]) == 1
    assert len(data["sources"]) == 1
    assert data["evidence_count"] == 1
    assert "text" not in data["sources"][0]

    with get_api_db_session() as db:
        run = db.get(ResearchRun, data["research_run_id"])

        assert run is not None
        assert run.status == "completed"
        assert run.query == question
        assert run.result_summary == (
            "The evidence supports the research claim."
        )
        assert run.error_message is None
        assert run.started_at is not None
        assert run.completed_at is not None


def test_research_api_returns_404_for_missing_workspace(client):
    app.dependency_overrides[get_research_graph] = (
        lambda: FakeResearchGraph()
    )

    response = client.post(
        "/api/v1/research/run",
        json={
            "workspace_id": 999999,
            "question": "Research a missing workspace.",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found."

    with get_api_db_session() as db:
        assert db.query(ResearchRun).count() == 0


def test_research_api_returns_503_when_graph_fails(client, test_workspace):
    app.dependency_overrides[get_research_graph] = (
        lambda: FakeResearchGraph(should_fail=True)
    )

    question = "Run a failing research workflow."

    response = client.post(
        "/api/v1/research/run",
        json={
            "workspace_id": test_workspace.id,
            "question": question,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Research workflow execution failed."
    )

    with get_api_db_session() as db:
        run = (
            db.query(ResearchRun)
            .filter(ResearchRun.query == question)
            .one()
        )

        assert run.status == "failed"
        assert run.result_summary is None
        assert run.error_message == (
            "Research workflow execution failed."
        )
        assert run.started_at is not None
        assert run.completed_at is not None


def test_research_api_rejects_invalid_request(client):
    app.dependency_overrides[get_research_graph] = (
        lambda: FakeResearchGraph(should_fail=True)
    )

    response = client.post(
        "/api/v1/research/run",
        json={
            "workspace_id": 0,
            "question": "",
        },
    )

    assert response.status_code == 422

    with get_api_db_session() as db:
        assert db.query(ResearchRun).count() == 0
