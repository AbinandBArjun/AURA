from unittest.mock import patch

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

    with patch(
        "app.services.search_service.QdrantService"
    ) as mock_qdrant:

        mock_qdrant.return_value.search.return_value = mock_results

        service = SearchService()

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
    assert results[0]["score"] == 0.85


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


def test_search_api_success(db_session, test_workspace):
    from app.db.session import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
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

            client = TestClient(app)

            response = client.post(
                "/api/v1/search",
                json={
                    "workspace_id": test_workspace.id,
                    "query": "Research findings",
                    "limit": 5,
                },
            )

        assert response.status_code == 200
        assert response.json()["result_count"] == 1
        assert response.json()["results"][0]["filename"] == "test-paper.pdf"

    finally:
        app.dependency_overrides.pop(get_db, None)


def test_search_api_rejects_invalid_request():
    client = TestClient(app)

    response = client.post(
        "/api/v1/search",
        json={
            "workspace_id": 0,
            "query": "",
            "limit": 100,
        },
    )

    assert response.status_code == 422
