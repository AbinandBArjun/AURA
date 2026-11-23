import json
import math
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models import Document, DocumentChunk
from app.retrieval.bm25_retriever import BM25Retriever

from evaluation.run_bm25_performance import (
    DATASET_PATH,
    MEASURED_ROUNDS,
    TOP_K,
    WARMUP_ROUNDS,
    percentile,
    profile_search,
    summarize,
)


OUTPUT_PATH = (
    Path(__file__).parent
    / "results"
    / "bm25_performance_optimized.json"
)


def assert_same_results(actual, expected, query_id):
    actual_ids = [item["chunk_id"] for item in actual]
    expected_ids = [item["chunk_id"] for item in expected]

    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Chunk ranking mismatch for {query_id}: "
            f"{actual_ids} != {expected_ids}"
        )

    for actual_item, expected_item in zip(actual, expected):
        if not math.isclose(
            actual_item["score"],
            expected_item["score"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Score mismatch for {query_id}, "
                f"chunk {actual_item['chunk_id']}"
            )


def run_benchmark():
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    workspace_id = dataset["workspace_id"]
    queries = dataset["queries"]

    retriever = BM25Retriever()

    measurements = {
        "cold_search": [],
        "warm_search": [],
    }

    per_query = {}

    print("AURA BM25 Optimized Performance Benchmark")
    print("=" * 50)
    print("Workspace:", workspace_id)
    print("Queries:", len(queries))
    print("Warm-up rounds:", WARMUP_ROUNDS)
    print("Measured rounds:", MEASURED_ROUNDS)

    with SessionLocal() as db:
        for item in queries:
            query_id = item["query_id"]
            query = item["query"]

            print(f"\nProcessing {query_id}")

            reference, _ = profile_search(
                db,
                retriever,
                workspace_id,
                query,
                TOP_K,
            )

            query_measurements = {
                "cold_search": [],
                "warm_search": [],
            }

            # Warm up both paths without retaining an old index.
            for _ in range(WARMUP_ROUNDS):
                retriever.index_cache.invalidate(workspace_id)

                cold_results = retriever.search(
                    db, workspace_id, query, TOP_K
                )
                assert_same_results(cold_results, reference, query_id)

                warm_results = retriever.search(
                    db, workspace_id, query, TOP_K
                )
                assert_same_results(warm_results, reference, query_id)

            # Measure cold and warm production searches separately.
            for _ in range(MEASURED_ROUNDS):
                retriever.index_cache.invalidate(workspace_id)

                start = time.perf_counter()
                cold_results = retriever.search(
                    db, workspace_id, query, TOP_K
                )
                cold_ms = (time.perf_counter() - start) * 1000

                assert_same_results(cold_results, reference, query_id)

                start = time.perf_counter()
                warm_results = retriever.search(
                    db, workspace_id, query, TOP_K
                )
                warm_ms = (time.perf_counter() - start) * 1000

                assert_same_results(warm_results, reference, query_id)

                query_measurements["cold_search"].append(cold_ms)
                query_measurements["warm_search"].append(warm_ms)

                measurements["cold_search"].append(cold_ms)
                measurements["warm_search"].append(warm_ms)

            per_query[query_id] = {
                "category": item["category"],
                "measurements": {
                    stage: summarize(values)
                    for stage, values in query_measurements.items()
                },
            }

            print(
                "  Cold median:",
                f"{statistics.median(query_measurements['cold_search']):.3f} ms",
            )
            print(
                "  Warm median:",
                f"{statistics.median(query_measurements['warm_search']):.3f} ms",
            )

    with SessionLocal() as db:
        document_count = db.scalar(
            select(func.count(Document.id))
            .where(Document.workspace_id == workspace_id)
        )

        chunk_counts = db.execute(
            select(
                func.count(DocumentChunk.id),
                func.count(DocumentChunk.id).filter(
                    DocumentChunk.retrievable.is_(True)
                ),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.workspace_id == workspace_id)
        ).one()

    results = {
        stage: summarize(values)
        for stage, values in measurements.items()
    }

    cold_median = results["cold_search"]["median_ms"]
    warm_median = results["warm_search"]["median_ms"]

    output = {
        "benchmark": "AURA BM25 Optimized Performance",
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["dataset_version"],
        "workspace_id": workspace_id,
        "query_count": len(queries),
        "warmup_rounds": WARMUP_ROUNDS,
        "measured_rounds": MEASURED_ROUNDS,
        "top_k": TOP_K,
        "corpus": {
            "document_count": document_count,
            "total_chunks": chunk_counts[0],
            "retrievable_chunks": chunk_counts[1],
        },
        "bm25_library_version": version("rank-bm25"),
        "results": results,
        "warm_to_cold_median_speedup": (
            cold_median / warm_median if warm_median > 0 else None
        ),
        "per_query": per_query,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("AGGREGATE RESULTS")

    for stage, metrics in results.items():
        print(
            f"{stage:15} "
            f"mean={metrics['mean_ms']:.3f} ms  "
            f"median={metrics['median_ms']:.3f} ms  "
            f"p95={metrics['p95_ms']:.3f} ms"
        )

    print(
        "\nWarm-to-cold median speedup:",
        f"{output['warm_to_cold_median_speedup']:.2f}x",
    )
    print("\nResults saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_benchmark()
