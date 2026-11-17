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
                    "chunk_id": 10,
                    "text": "Hybrid retrieval combines dense and sparse search.",
                }
            ]
        }


def test_research_graph_executes_planner_and_researcher():
    graph = build_research_graph(
        planner=MockPlanner(),
        researcher=MockResearcher(),
    )

    result = graph.invoke({
        "workspace_id": 1,
        "question": "Explain hybrid RAG",
    })

    assert len(result["research_plan"]) == 3
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["document_id"] == 1


def test_research_graph_preserves_initial_state():
    graph = build_research_graph(
        planner=MockPlanner(),
        researcher=MockResearcher(),
    )

    result = graph.invoke({
        "workspace_id": 42,
        "question": "Explain hybrid RAG",
    })

    assert result["workspace_id"] == 42
    assert result["question"] == "Explain hybrid RAG"


def test_research_graph_preserves_evidence_metadata():
    graph = build_research_graph(
        planner=MockPlanner(),
        researcher=MockResearcher(),
    )

    result = graph.invoke({
        "workspace_id": 1,
        "question": "Explain hybrid RAG",
    })

    evidence = result["evidence"][0]

    assert evidence["chunk_id"] == 10
    assert evidence["text"] == (
        "Hybrid retrieval combines dense and sparse search."
    )
