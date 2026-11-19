import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import Document
from app.services.search_service import SearchService


def test_search_service_returns_document_metadata(
    db_session,
    test_workspace,
):
    document = Document(
        workspace_id=test_workspace.id,
        filename="test-paper.pdf",
        content_type="application/pdf",
        size_bytes=100,
        checksum_sha256="test-search-checksum",
        storage_key="test-paper.pdf",
        status="processed",
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    mock_results = [
        {
            "document_id": document.id,
            "chunk_id": 10,
            "text": "Research findings about RAG.",
            "page_start": 1,
            "page_end": 2,
            "score": 0.85,
        }
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, results, limit: [
        {
            **result,
            "rrf_score": result["rrf_score"],
            "rerank_score": 0.92,
        }
        for result in results[:limit]
    ]

    with patch(
        "app.services.search_service.QdrantService"
    ) as mock_qdrant:

        mock_qdrant.return_value.search.return_value = mock_results

        service = SearchService(reranker=mock_reranker)

        results = service.search(
            db=db_session,
            workspace_id=test_workspace.id,
            query="What are the research findings?",
            limit=5,
        )

    assert len(results) == 1
    assert results[0]["filename"] == "test-paper.pdf"
    assert results[0]["chunk_id"] == 10
    assert results[0]["page_start"] == 1
    assert results[0]["score"] == pytest.approx(0.92)


def test_search_service_rejects_missing_workspace(db_session):
    service = SearchService()

    try:
        service.search(
            db=db_session,
            workspace_id=99999,
            query="Test query",
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Workspace not found."


def test_search_api_success(client, current_workspace):
    with patch(
        "app.api.routes.search.SearchService"
    ) as mock_service:

        mock_service.return_value.search.return_value = [
            {
                "document_id": 1,
                "filename": "test-paper.pdf",
                "chunk_id": 10,
                "text": "Research findings about RAG.",
                "page_start": 1,
                "page_end": 2,
                "score": 0.85,
            }
        ]

        response = client.post(
            "/api/v1/search",
            json={
                "query": "Research findings",
                "limit": 5,
            },
        )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1
    assert response.json()["results"][0]["filename"] == "test-paper.pdf"

    search_call = mock_service.return_value.search.call_args

    assert search_call.kwargs["workspace_id"] == current_workspace.id
    assert search_call.kwargs["query"] == "Research findings"
    assert search_call.kwargs["limit"] == 5


def test_search_api_rejects_invalid_request(client):
    response = client.post(
        "/api/v1/search",
        json={
            "query": "",
            "limit": 100,
        },
    )

    assert response.status_code == 422


def test_search_service_returns_bm25_results_without_dense_matches(
    db_session,
    test_workspace,
):
    document = Document(
        workspace_id=test_workspace.id,
        filename="sparse-paper.pdf",
        content_type="application/pdf",
        size_bytes=100,
        checksum_sha256="sparse-search-checksum",
        storage_key="sparse-paper.pdf",
        status="processed",
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    qdrant = MagicMock()
    qdrant.search.return_value = []

    bm25 = MagicMock()
    bm25.search.return_value = [
        {
            "document_id": document.id,
            "filename": document.filename,
            "chunk_id": 25,
            "text": "Artificial intelligence changes labor markets.",
            "page_start": 4,
            "page_end": 5,
            "section": "ai_labor",
            "score": 2.5,
        }
    ]

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, results, limit: [
        {
            **result,
            "rerank_score": 0.91,
        }
        for result in results[:limit]
    ]

    service = SearchService(
        qdrant_service=qdrant,
        bm25_retriever=bm25,
        reranker=reranker,
    )

    results = service.search(
        db=db_session,
        workspace_id=test_workspace.id,
        query="artificial intelligence labor markets",
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == 25
    assert results[0]["filename"] == "sparse-paper.pdf"
    assert results[0]["score"] == pytest.approx(0.91)

    qdrant.search.assert_called_once()
    bm25.search.assert_called_once()


def test_search_service_uses_reranked_order(
    db_session,
    test_workspace,
):
    document = Document(
        workspace_id=test_workspace.id,
        filename="reranking-paper.pdf",
        content_type="application/pdf",
        size_bytes=100,
        checksum_sha256="reranking-search-checksum",
        storage_key="reranking-paper.pdf",
        status="processed",
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    qdrant = MagicMock()
    qdrant.search.return_value = [
        {
            "document_id": document.id,
            "chunk_id": 1,
            "text": "General artificial intelligence discussion.",
            "page_start": 1,
            "page_end": 2,
            "score": 0.90,
        }
    ]

    bm25 = MagicMock()
    bm25.search.return_value = [
        {
            "document_id": document.id,
            "filename": document.filename,
            "chunk_id": 2,
            "text": "Specific evidence about AI labor markets.",
            "page_start": 3,
            "page_end": 4,
            "score": 2.5,
        }
    ]

    reranker = MagicMock()

    def mock_rerank(query, results, limit):
        ranked = sorted(
            results,
            key=lambda item: item["chunk_id"],
            reverse=True,
        )

        return [
            {
                **item,
                "rerank_score": 0.95 - index * 0.1,
            }
            for index, item in enumerate(ranked[:limit])
        ]

    reranker.rerank.side_effect = mock_rerank

    service = SearchService(
        qdrant_service=qdrant,
        bm25_retriever=bm25,
        reranker=reranker,
    )

    results = service.search(
        db=db_session,
        workspace_id=test_workspace.id,
        query="AI labor markets",
        limit=5,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == 2
    assert results[1]["chunk_id"] == 1

    reranker.rerank.assert_called_once()
