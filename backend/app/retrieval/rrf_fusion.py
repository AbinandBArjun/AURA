from typing import Any


class RRFFusion:
    """Combine ranked retrieval results using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError("RRF constant k must be positive.")

        self.k = k

    def fuse(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("Fusion limit must be positive.")

        fused: dict[int, dict[str, Any]] = {}

        ranked_lists = [
            ("dense", dense_results),
            ("sparse", sparse_results),
        ]

        for source, results in ranked_lists:
            seen_in_list = set()

            for rank, result in enumerate(results, start=1):
                chunk_id = result.get("chunk_id")

                if chunk_id is None:
                    raise ValueError(
                        f"Retrieval result is missing chunk_id: {source}"
                    )

                if chunk_id in seen_in_list:
                    continue

                seen_in_list.add(chunk_id)

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **result,
                        "rrf_score": 0.0,
                        "dense_rank": None,
                        "sparse_rank": None,
                    }

                fused[chunk_id]["rrf_score"] += 1.0 / (self.k + rank)
                fused[chunk_id][f"{source}_rank"] = rank

        ranked_results = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["chunk_id"],
            ),
        )

        return ranked_results[:limit]
