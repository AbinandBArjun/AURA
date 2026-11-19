from app.api.routes import rag as rag_route


def test_rag_answer_uses_server_resolved_workspace(
    client,
    current_workspace,
    monkeypatch,
):
    captured = {}

    class FakeRAGService:
        def __init__(self, search_service):
            captured["search_service"] = search_service

        def answer(self, workspace_id, question, limit):
            captured["workspace_id"] = workspace_id
            captured["question"] = question
            captured["limit"] = limit

            return {
                "question": question,
                "answer": "Retrieval-augmented generation combines retrieval and generation. [S1]",
                "citations": [],
                "evidence_count": 1,
                "claims": [],
            }

    monkeypatch.setattr(
        rag_route,
        "SearchService",
        lambda: "fake-search-service",
    )
    monkeypatch.setattr(
        rag_route,
        "RAGService",
        FakeRAGService,
    )

    response = client.post(
        "/api/v1/rag/answer",
        json={
            "question": "What is retrieval-augmented generation?",
            "limit": 7,
        },
    )

    assert response.status_code == 200
    assert response.json()["question"] == (
        "What is retrieval-augmented generation?"
    )
    assert response.json()["evidence_count"] == 1

    assert captured["workspace_id"] == current_workspace.id
    assert captured["question"] == (
        "What is retrieval-augmented generation?"
    )
    assert captured["limit"] == 7
    assert captured["search_service"] == "fake-search-service"


def test_rag_answer_rejects_client_supplied_workspace_id(
    client,
    current_workspace,
):
    response = client.post(
        "/api/v1/rag/answer",
        json={
            "workspace_id": current_workspace.id,
            "question": "What is retrieval-augmented generation?",
            "limit": 5,
        },
    )

    assert response.status_code == 422


def test_rag_answer_returns_404_for_value_error(
    client,
    current_workspace,
    monkeypatch,
):
    class FakeRAGService:
        def __init__(self, search_service):
            pass

        def answer(self, workspace_id, question, limit):
            raise ValueError("No indexed evidence found.")

    monkeypatch.setattr(
        rag_route,
        "SearchService",
        lambda: "fake-search-service",
    )
    monkeypatch.setattr(
        rag_route,
        "RAGService",
        FakeRAGService,
    )

    response = client.post(
        "/api/v1/rag/answer",
        json={"question": "Find evidence about RAG."},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "No indexed evidence found."


def test_rag_answer_returns_503_for_rag_error(
    client,
    current_workspace,
    monkeypatch,
):
    class FakeRAGService:
        def __init__(self, search_service):
            pass

        def answer(self, workspace_id, question, limit):
            raise rag_route.RAGError("The language model is unavailable.")

    monkeypatch.setattr(
        rag_route,
        "SearchService",
        lambda: "fake-search-service",
    )
    monkeypatch.setattr(
        rag_route,
        "RAGService",
        FakeRAGService,
    )

    response = client.post(
        "/api/v1/rag/answer",
        json={"question": "Find evidence about RAG."},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The language model is unavailable."
    )
