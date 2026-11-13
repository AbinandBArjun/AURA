import pytest

from evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
)


@pytest.fixture
def relevance():
    return {
        "101": 3,
        "102": 2,
        "103": 1,
        "104": 0,
    }


def test_recall_at_k(relevance):
    retrieved = [101, 104, 102]

    assert recall_at_k(retrieved, relevance, 2) == pytest.approx(
        1 / 3
    )


def test_reciprocal_rank(relevance):
    retrieved = [104, 102, 101]

    assert reciprocal_rank(retrieved, relevance) == pytest.approx(
        0.5
    )


def test_reciprocal_rank_when_no_relevant_results(relevance):
    assert reciprocal_rank([104], relevance) == 0.0


def test_dcg_at_k(relevance):
    score = dcg_at_k([101, 102], relevance, 2)

    expected = 7 + (3 / 1.5849625007)

    assert score == pytest.approx(expected)


def test_ndcg_at_k_for_ideal_ranking(relevance):
    assert ndcg_at_k([101, 102, 103], relevance, 3) == pytest.approx(
        1.0
    )


def test_recall_rejects_invalid_k(relevance):
    with pytest.raises(ValueError):
        recall_at_k([101], relevance, 0)


def test_dcg_rejects_invalid_k(relevance):
    with pytest.raises(ValueError):
        dcg_at_k([101], relevance, 0)


def test_precision_at_k():
    retrieved = ["101", "102", "103", "104", "105"]

    relevant = {
        "101": 3,
        "102": 2,
        "103": 0,
        "104": 1,
        "105": 0,
    }

    assert precision_at_k(retrieved, relevant, 5) == 0.6


def test_precision_at_k_with_integer_ids():
    retrieved = [101, 102, 103]

    relevant = {
        "101": 3,
        "102": 0,
        "103": 2,
    }

    assert precision_at_k(retrieved, relevant, 3) == 2 / 3


def test_precision_at_k_when_no_results():
    assert precision_at_k([], {"101": 3}, 5) == 0.0


def test_precision_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["101"], {"101": 3}, 0)
