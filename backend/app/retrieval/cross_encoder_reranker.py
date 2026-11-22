import math
from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved passages using query-passage relevance scores."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model=None,
        batch_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size

    def _get_model(self):
        # Lazy loading avoids downloading the model during application startup.
        if self.model is None:
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
                device="cpu",
            )

        return self.model

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Reranking query cannot be empty.")

        if limit <= 0:
            raise ValueError("Reranking limit must be positive.")

        if not results:
            return []

        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Each reranker result must be a dictionary.")

            text = result.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Each reranker result must contain valid non-empty text."
                )

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self._get_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        try:
            score_count = len(scores)
        except TypeError as exc:
            raise ValueError(
                "Reranker returned an unexpected number of scores."
            ) from exc

        if score_count != len(results):
            raise ValueError("Reranker returned an unexpected number of scores.")

        validated_scores = []

        for score in scores:
            if isinstance(score, (str, bytes, bool)):
                raise ValueError(
                    "Reranker must return finite numeric scores."
                )

            try:
                numeric_score = float(score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Reranker must return finite numeric scores."
                ) from exc

            if not math.isfinite(numeric_score):
                raise ValueError(
                    "Reranker must return finite numeric scores."
                )

            validated_scores.append(numeric_score)

        ranked_results = []

        for original_index, (result, score) in enumerate(
            zip(results, validated_scores)
        ):
            item = dict(result)
            item["rrf_score"] = result.get("rrf_score", result.get("score"))
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            item["_original_index"] = original_index
            ranked_results.append(item)

        ranked_results.sort(
            key=lambda item: (
                -item["rerank_score"],
                item["_original_index"],
            )
        )

        for item in ranked_results:
            item.pop("_original_index", None)

        return ranked_results[:limit]
