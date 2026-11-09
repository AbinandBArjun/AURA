import pytest

from app.retrieval.rrf_fusion import RRFFusion


@pytest.fixture
def fusion():
    return RRFFusion(k=60)


def test_rrf_combines_dense_and_sparse_results(fusion):
    dense = [
        {"chunk_id": 1, "text": "Dense result 1"},
        {"chunk_id": 2, "text": "Dense result 2"},
    ]

    sparse = [
        {"chunk_id": 2, "text": "Sparse result 2"},
        {"chunk_id": 3, "text": "Sparse result 3"},
    ]

    results = fusion.fuse(dense, sparse)

    assert [item["chunk_id"] for item in results] == [2, 1, 3]


def test_rrf_rewards_results_found_by_both_retrievers(fusion):
    dense = [{"chunk_id": 1}, {"chunk_id": 2}]
    sparse = [{"chunk_id": 2}, {"chunk_id": 3}]

    results = fusion.fuse(dense, sparse)

    top_result = results[0]

    assert top_result["chunk_id"] == 2
    assert top_result["dense_rank"] == 2
    assert top_result["sparse_rank"] == 1


def test_rrf_score_uses_correct_formula(fusion):
    dense = [{"chunk_id": 1}]
    sparse = [{"chunk_id": 1}]

    results = fusion.fuse(dense, sparse)

    expected = (1 / 61) + (1 / 61)

    assert results[0]["rrf_score"] == pytest.approx(expected)


def test_rrf_deduplicates_within_each_retrieval_list(fusion):
    dense = [
        {"chunk_id": 1},
        {"chunk_id": 1},
        {"chunk_id": 2},
    ]

    results = fusion.fuse(dense, [])

    assert len(results) == 2
    assert results[0]["chunk_id"] == 1
    assert results[0]["dense_rank"] == 1


def test_rrf_respects_result_limit(fusion):
    dense = [
        {"chunk_id": 1},
        {"chunk_id": 2},
        {"chunk_id": 3},
    ]

    results = fusion.fuse(dense, [], limit=2)

    assert len(results) == 2


def test_rrf_rejects_invalid_limit(fusion):
    with pytest.raises(ValueError, match="Fusion limit must be positive"):
        fusion.fuse([], [], limit=0)


def test_rrf_rejects_missing_chunk_id(fusion):
    with pytest.raises(ValueError, match="missing chunk_id"):
        fusion.fuse([{"text": "Missing ID"}], [])


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError, match="RRF constant k must be positive"):
        RRFFusion(k=0)
