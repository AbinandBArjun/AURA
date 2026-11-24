import os
from pathlib import Path

import pytest

from app.embeddings.embedding_service import EmbeddingService
from prepare_embedding_model import validate_model


pytestmark = pytest.mark.real_embedding_model


@pytest.fixture(scope="module")
def embedding_service():
    local_model = os.environ.get("AURA_TEST_EMBEDDING_MODEL")
    if not local_model:
        pytest.fail("Set AURA_TEST_EMBEDDING_MODEL to a prepared model directory.", pytrace=False)

    model_path = validate_model(Path(local_model))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(EmbeddingService, "MODEL_NAME", str(model_path))
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

