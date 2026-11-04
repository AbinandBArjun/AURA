import pytest

from app.embeddings.embedding_service import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def test_single_text_embedding(embedding_service):
    vector = embedding_service.embed_text(
        "Retrieval augmented generation uses external knowledge."
    )

    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)


def test_batch_embeddings(embedding_service):
    texts = [
        "Artificial intelligence research",
        "Machine learning models",
        "Vector databases",
    ]

    vectors = embedding_service.embed_texts(texts)

    assert len(vectors) == 3
    assert all(len(vector) == 384 for vector in vectors)


def test_empty_text_is_rejected(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.embed_text("   ")


def test_empty_batch_returns_empty_list(embedding_service):
    assert embedding_service.embed_texts([]) == []

