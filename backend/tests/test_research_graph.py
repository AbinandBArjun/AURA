from app.agents.graph import build_research_graph


class MockPlanner:
    def __call__(self, state):
        return {
            "research_plan": [
                "Understand dense retrieval",
                "Understand sparse retrieval",
                "Compare hybrid retrieval methods",
            ]
        }


class MockResearcher:
    def __call__(self, state):
        assert len(state["research_plan"]) == 3

        return {
            "evidence": [
                {
                    "document_id": 1,
                    "filename": "research.pdf",
                    "chunk_id": 10,
                    "text": "Hybrid retrieval combines dense and sparse search.",
                    "page_start": 1,
                    "page_end": 2,
                }
            ]
        }


class MockAnalyst:
    def __call__(self, state):
        assert state["evidence"][0]["citation_id"] == "S1"

        return {
            "findings": "Hybrid retrieval combines dense and sparse search.",
            "claims": [
                {
                    "claim_id": "C1",
                    "claim_text": (
                        "Hybrid retrieval combines dense and sparse search."
                    ),
                    "evidence_ids": ["S1"],
                }
            ],
        }


class MockFactChecker:
    def __call__(self, state):
        assert state["claims"][0]["claim_id"] == "C1"

        return {
            "verification_results": [
                {
                    "claim_id": "C1",
                    "status": "supported",
                    "rationale": "The evidence directly supports the claim.",
                    "evidence_ids": ["S1"],
                }
            ]
        }


class MockReportGenerator:
    def __call__(self, state):
        assert state["verification_results"][0]["status"] == "supported"

        return {"report": "# Research Report\n\nMock report."}


def build_mock_graph():
    return build_research_graph(
        planner=MockPlanner(),
        researcher=MockResearcher(),
        analyst=MockAnalyst(),
        fact_checker=MockFactChecker(),
        report_generator=MockReportGenerator(),
    )


def test_research_graph_executes_complete_workflow():
    graph = build_mock_graph()

    result = graph.invoke({
        "workspace_id": 1,
        "question": "Explain hybrid RAG",
    })

    assert len(result["research_plan"]) == 3
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["citation_id"] == "S1"
    assert result["findings"]
    assert result["claims"][0]["claim_id"] == "C1"
    assert result["verification_results"][0]["status"] == "supported"
    assert result["report"] == "# Research Report\n\nMock report."


def test_research_graph_preserves_initial_state():
    graph = build_mock_graph()

    result = graph.invoke({
        "workspace_id": 42,
        "question": "Explain hybrid RAG",
    })

    assert result["workspace_id"] == 42
    assert result["question"] == "Explain hybrid RAG"


def test_research_graph_preserves_evidence_metadata():
    graph = build_mock_graph()

    result = graph.invoke({
        "workspace_id": 1,
        "question": "Explain hybrid RAG",
    })

    evidence = result["evidence"][0]

    assert evidence["document_id"] == 1
    assert evidence["filename"] == "research.pdf"
    assert evidence["chunk_id"] == 10
    assert evidence["page_start"] == 1
    assert evidence["page_end"] == 2
    assert evidence["text"] == (
        "Hybrid retrieval combines dense and sparse search."
    )
    assert evidence["citation_id"] == "S1"
