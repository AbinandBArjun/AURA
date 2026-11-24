import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app.db.session import SessionLocal
from app.services.search_service import SearchService
from evaluation.metrics.retrieval_metrics import (
    RANKING_METRICS,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)

DATASET_PATH = (
    Path(__file__).parent
    / "datasets"
    / "retrieval_benchmark.json"
)

OUTPUT_PATH = (
    Path(__file__).parent
    / "results"
    / "retrieval_benchmark_results.json"
)

TOP_K = 5

RETRIEVAL_CONFIGS = [
    {
        "name": "dense",
        "mode": "dense",
        "candidate_pool_size": None,
    },
    {
        "name": "hybrid",
        "mode": "hybrid",
        "candidate_pool_size": None,
    },
    {
        "name": "hybrid_rerank_5",
        "mode": "hybrid_rerank",
        "candidate_pool_size": 5,
    },
    {
        "name": "hybrid_rerank_10",
        "mode": "hybrid_rerank",
        "candidate_pool_size": 10,
    },
    {
        "name": "hybrid_rerank_20",
        "mode": "hybrid_rerank",
        "candidate_pool_size": 20,
    },
]

METRICS = list(RANKING_METRICS)

LATENCY_METRICS = [
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
]


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_query(
    search_service,
    db,
    workspace_id,
    item,
    mode,
    candidate_pool_size=None,
):
    search_kwargs = {
        "db": db,
        "workspace_id": workspace_id,
        "query": item["query"],
        "limit": TOP_K,
        "mode": mode,
    }

    if candidate_pool_size is not None:
        search_kwargs["candidate_pool_size"] = candidate_pool_size

    start_time = time.perf_counter()
    results = search_service.search(**search_kwargs)
    latency_ms = (time.perf_counter() - start_time) * 1000

    retrieved_ids = [
        str(result["chunk_id"])
        for result in results
    ]

    relevant_chunks = item["relevant_chunks"]

    return {
        "query_id": item["query_id"],
        "category": item["category"],
        "retrieved_chunk_ids": retrieved_ids,
        "latency_ms": round(latency_ms, 3),
        "precision_at_5": precision_at_k(
            retrieved_ids, relevant_chunks, TOP_K
        ),
        "recall_at_5": recall_at_k(
            retrieved_ids, relevant_chunks, TOP_K
        ),
        "mrr": reciprocal_rank(
            retrieved_ids, relevant_chunks
        ),
        "ndcg_at_5": ndcg_at_k(
            retrieved_ids, relevant_chunks, TOP_K
        ),
    }


def mean_metric(results, metric):
    values = [result[metric] for result in results]
    return statistics.mean(values) if values else 0.0


def summarize_results(results):
    return {
        metric: mean_metric(results, metric)
        for metric in METRICS
    }


def summarize_latency(results):
    values = sorted(
        result["latency_ms"]
        for result in results
    )

    if not values:
        return {
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    p95 = statistics.quantiles(
        values,
        n=100,
        method="inclusive",
    )[94]

    return {
        "mean_latency_ms": statistics.mean(values),
        "median_latency_ms": statistics.median(values),
        "p95_latency_ms": p95,
    }


def group_by_category(query_results):
    categories = defaultdict(list)

    for result in query_results:
        categories[result["category"]].append(result)

    return {
        category: {
            "query_count": len(results),
            **summarize_results(results),
        }
        for category, results in sorted(categories.items())
    }


def run_benchmark(output_path=OUTPUT_PATH):
    dataset = load_dataset()
    workspace_id = dataset["workspace_id"]

    search_service = SearchService()

    first_query = dataset["queries"][0]

    print("\nWarming up retrieval models...")

    with SessionLocal() as db:
        search_service.search(
            db=db,
            workspace_id=workspace_id,
            query=first_query["query"],
            limit=TOP_K,
            mode="dense",
        )

        search_service.search(
            db=db,
            workspace_id=workspace_id,
            query=first_query["query"],
            limit=TOP_K,
            mode="hybrid_rerank",
            candidate_pool_size=20,
        )

    print("Warm-up complete.")

    all_results = {}

    with SessionLocal() as db:
        for config in RETRIEVAL_CONFIGS:
            name = config["name"]
            mode = config["mode"]
            candidate_pool_size = config["candidate_pool_size"]

            print(f"\nEvaluating: {name}")

            if candidate_pool_size is not None:
                print(
                    f"  Reranker candidate pool: "
                    f"{candidate_pool_size}"
                )

            query_results = []

            for item in dataset["queries"]:
                result = evaluate_query(
                    search_service=search_service,
                    db=db,
                    workspace_id=workspace_id,
                    item=item,
                    mode=mode,
                    candidate_pool_size=candidate_pool_size,
                )

                query_results.append(result)

                print(
                    f"  {item['query_id']}: "
                    f"Precision@5={result['precision_at_5']:.3f}, "
                    f"Recall@5={result['recall_at_5']:.3f}, "
                    f"MRR={result['mrr']:.3f}, "
                    f"nDCG@5={result['ndcg_at_5']:.3f}"
                )

            category_results = group_by_category(query_results)

            all_results[name] = {
                "mode": mode,
                "candidate_pool_size": candidate_pool_size,
                **summarize_results(query_results),
                **summarize_latency(query_results),
                "per_category": category_results,
                "per_query": query_results,
            }

    print("\n========== AGGREGATE RESULTS ==========")

    for name, metrics in all_results.items():
        print(f"\n{name}")

        for metric in METRICS:
            print(f"  {metric}: {metrics[metric]:.4f}")

        print("\n  LATENCY")

        for metric in LATENCY_METRICS:
            print(f"  {metric}: {metrics[metric]:.2f}")

        print("\n  CATEGORY RESULTS")

        for category, category_metrics in metrics["per_category"].items():
            print(
                f"  {category} "
                f"(n={category_metrics['query_count']}): "
                f"Precision@5={category_metrics['precision_at_5']:.3f}, "
                f"Recall@5={category_metrics['recall_at_5']:.3f}, "
                f"MRR={category_metrics['mrr']:.3f}, "
                f"nDCG@5={category_metrics['ndcg_at_5']:.3f}"
            )

    output = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["dataset_version"],
        "workspace_id": workspace_id,
        "top_k": TOP_K,
        "query_count": len(dataset["queries"]),
        "retrieval_configs": RETRIEVAL_CONFIGS,
        "results": all_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nResults saved to: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the retrieval benchmark.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Results path. Use a separate file when checking against the tracked baseline.",
    )
    args = parser.parse_args(argv)
    run_benchmark(args.output)


if __name__ == "__main__":
    main()
