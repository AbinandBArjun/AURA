import math


RANKING_METRICS = ("precision_at_5", "recall_at_5", "mrr", "ndcg_at_5")


def _normalise_relevance(
    relevant_chunks: dict[str, int],
) -> dict[str, int]:
    return {
        str(chunk_id): grade
        for chunk_id, grade in relevant_chunks.items()
    }


def recall_at_k(
    retrieved_ids: list[int | str],
    relevant_chunks: dict[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    relevance = _normalise_relevance(relevant_chunks)

    relevant_ids = {
        chunk_id
        for chunk_id, grade in relevance.items()
        if grade > 0
    }

    if not relevant_ids:
        return 0.0

    retrieved = {str(chunk_id) for chunk_id in retrieved_ids[:k]}

    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[int | str],
    relevant_chunks: dict[str, int],
) -> float:
    relevance = _normalise_relevance(relevant_chunks)

    relevant_ids = {
        chunk_id
        for chunk_id, grade in relevance.items()
        if grade > 0
    }

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if str(chunk_id) in relevant_ids:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved_ids: list[int | str],
    relevant_chunks: dict[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    relevance = _normalise_relevance(relevant_chunks)
    score = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        grade = relevance.get(str(chunk_id), 0)
        gain = (2 ** grade) - 1
        discount = math.log2(rank + 1)
        score += gain / discount

    return score


def ndcg_at_k(
    retrieved_ids: list[int | str],
    relevant_chunks: dict[str, int],
    k: int,
) -> float:
    actual_dcg = dcg_at_k(retrieved_ids, relevant_chunks, k)

    relevance = _normalise_relevance(relevant_chunks)

    ideal_ids = sorted(
        relevance,
        key=lambda chunk_id: relevance[chunk_id],
        reverse=True,
    )

    ideal_dcg = dcg_at_k(ideal_ids, relevance, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def precision_at_k(
    retrieved_ids: list[int | str],
    relevant_chunks: dict[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    relevance = _normalise_relevance(relevant_chunks)

    if not retrieved_ids:
        return 0.0

    retrieved = [
        str(chunk_id)
        for chunk_id in retrieved_ids[:k]
    ]

    relevant_count = sum(
        1
        for chunk_id in retrieved
        if relevance.get(chunk_id, 0) > 0
    )

    return relevant_count / k
