from unittest.mock import MagicMock

import pytest

from app.retrieval.cross_encoder_reranker import CrossEncoderReranker


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0.2, 0.9, 0.5]
    return model


@pytest.fixture
def reranker(mock_model):
    return CrossEncoderReranker(model=mock_model)


@pytest.fixture
def sample_results():
    return [
        {"chunk_id": 1, "text": "General AI information", "score": 0.04},
        {"chunk_id": 2, "text": "AI and labor market evidence", "score": 0.03},
        {"chunk_id": 3, "text": "AI productivity research", "score": 0.02},
    ]


def test_reranks_by_cross_encoder_score(
    reranker,
    sample_results,
):
    results = reranker.rerank(
        query="What is the impact of AI on labor markets?",
        results=sample_results,
        limit=3,
    )

    assert [item["chunk_id"] for item in results] == [2, 3, 1]


def test_preserves_original_rrf_score(
    reranker,
    sample_results,
):
    results = reranker.rerank(
        query="AI labor markets",
        results=sample_results,
    )

    result = next(item for item in results if item["chunk_id"] == 2)

    assert result["rrf_score"] == 0.03
    assert result["rerank_score"] == pytest.approx(0.9)
    assert result["score"] == pytest.approx(0.9)


def test_respects_result_limit(reranker, sample_results):
    results = reranker.rerank(
        query="AI",
        results=sample_results,
        limit=2,
    )

    assert len(results) == 2


def test_empty_results_return_empty_list(reranker):
    assert reranker.rerank("AI", []) == []


def test_empty_query_is_rejected(reranker):
    with pytest.raises(ValueError, match="Reranking query cannot be empty"):
        reranker.rerank(" ", [{"chunk_id": 1, "text": "AI"}])


def test_invalid_limit_is_rejected(reranker):
    with pytest.raises(ValueError, match="Reranking limit must be positive"):
        reranker.rerank("AI", [{"chunk_id": 1, "text": "AI"}], limit=0)


def test_predict_receives_query_passage_pairs(
    reranker,
    sample_results,
    mock_model,
):
    reranker.rerank("AI labor markets", sample_results)

    pairs = mock_model.predict.call_args.args[0]

    assert pairs[0] == (
        "AI labor markets",
        "General AI information",
    )
    assert len(pairs) == 3
