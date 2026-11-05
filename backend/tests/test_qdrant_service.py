import pytest

from app.vector_store.qdrant_service import QdrantService


class FakeEmbeddingService:
    """Deterministic embeddings for fast Qdrant integration tests."""

    def embed_text(self, text):
        if "machine learning" in text.lower():
            return [1.0] + [0.0] * 383

        if "cooking" in text.lower():
            return [0.0, 1.0] + [0.0] * 382

        return [0.0, 0.0, 1.0] + [0.0] * 381

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


@pytest.fixture
def qdrant_service():
    service = QdrantService(
        embedding_service=FakeEmbeddingService()
    )

    # Use a dedicated test collection.
    service.COLLECTION_NAME = "aura_test_chunks"

    service.ensure_collection()

    yield service

    if service.client.collection_exists(service.COLLECTION_NAME):
        service.client.delete_collection(service.COLLECTION_NAME)

    service.client.close()


def test_collection_creation(qdrant_service):
    assert qdrant_service.client.collection_exists(
        qdrant_service.COLLECTION_NAME
    )


def test_vector_insertion_and_search(qdrant_service):
    from types import SimpleNamespace

    chunks = [
        SimpleNamespace(
            id=101,
            document_id=1,
            chunk_index=0,
            text="Machine learning research methods",
            page_start=1,
            page_end=2,
            token_count=20,
        ),
        SimpleNamespace(
            id=102,
            document_id=2,
            chunk_index=0,
            text="Cooking recipes and ingredients",
            page_start=3,
            page_end=3,
            token_count=15,
        ),
    ]

    inserted = qdrant_service.upsert_chunks(
        chunks,
        workspace_id=1,
    )

    assert inserted == 2

    results = qdrant_service.search(
        query="machine learning",
        workspace_id=1,
        limit=5,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == 101
    assert results[0]["workspace_id"] == 1


def test_workspace_filtering(qdrant_service):
    from types import SimpleNamespace

    chunks = [
        SimpleNamespace(
            id=201,
            document_id=10,
            chunk_index=0,
            text="Machine learning research",
            page_start=1,
            page_end=1,
            token_count=10,
        ),
        SimpleNamespace(
            id=202,
            document_id=20,
            chunk_index=0,
            text="Machine learning research",
            page_start=1,
            page_end=1,
            token_count=10,
        ),
    ]

    qdrant_service.upsert_chunks(chunks[:1], workspace_id=1)
    qdrant_service.upsert_chunks(chunks[1:], workspace_id=2)

    results = qdrant_service.search(
        query="machine learning",
        workspace_id=1,
    )

    assert len(results) == 1
    assert results[0]["workspace_id"] == 1


def test_invalid_search_query_is_rejected(qdrant_service):
    with pytest.raises(ValueError):
        qdrant_service.search("", workspace_id=1)


def test_invalid_search_limit_is_rejected(qdrant_service):
    with pytest.raises(ValueError):
        qdrant_service.search(
            "machine learning",
            workspace_id=1,
            limit=0,
        )


def test_delete_stale_document_chunks(qdrant_service):
    from types import SimpleNamespace

    document_id = 101

    chunks = [
        SimpleNamespace(
            id=1,
            document_id=document_id,
            chunk_index=0,
            text="Current chunk one",
            page_start=1,
            page_end=1,
            token_count=3,
        ),
        SimpleNamespace(
            id=2,
            document_id=document_id,
            chunk_index=1,
            text="Current chunk two",
            page_start=1,
            page_end=1,
            token_count=3,
        ),
    ]

    qdrant_service.upsert_chunks(chunks, workspace_id=1)

    # Add a vector representing an obsolete chunk.
    stale_chunk = SimpleNamespace(
        id=3,
        document_id=document_id,
        chunk_index=2,
        text="Obsolete chunk",
        page_start=1,
        page_end=1,
        token_count=2,
    )

    qdrant_service.upsert_chunks([stale_chunk], workspace_id=1)

    qdrant_service.delete_stale_document_chunks(
        document_id=document_id,
        valid_chunk_ids=[1, 2],
    )

    results = qdrant_service.search(
        query="Current chunk",
        workspace_id=1,
        limit=10,
    )

    returned_ids = {result["chunk_id"] for result in results}

    assert returned_ids == {1, 2}


