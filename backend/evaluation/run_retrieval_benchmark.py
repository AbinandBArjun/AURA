import json
import statistics
from pathlib import Path

from app.db.session import SessionLocal
from app.services.search_service import SearchService
from evaluation.metrics.retrieval_metrics import (
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

RETRIEVAL_MODES = [
    "dense",
    "hybrid",
    "hybrid_rerank",
]


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_query(search_service, db, workspace_id, item, mode):
    results = search_service.search(
        db=db,
        workspace_id=workspace_id,
        query=item["query"],
        limit=TOP_K,
        mode=mode,
    )

    retrieved_ids = [
        str(result["chunk_id"])
        for result in results
    ]

    relevant_chunks = item["relevant_chunks"]

    return {
        "query_id": item["query_id"],
        "retrieved_chunk_ids": retrieved_ids,
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


def run_benchmark():
    dataset = load_dataset()
    workspace_id = dataset["workspace_id"]

    search_service = SearchService()
    all_results = {}

    with SessionLocal() as db:
        for mode in RETRIEVAL_MODES:
            print(f"\nEvaluating: {mode}")

            query_results = []

            for item in dataset["queries"]:
                result = evaluate_query(
                    search_service=search_service,
                    db=db,
                    workspace_id=workspace_id,
                    item=item,
                    mode=mode,
                )

                query_results.append(result)

                print(
                    f"  {item['query_id']}: "
                    f"Recall@5={result['recall_at_5']:.3f}, "
                    f"MRR={result['mrr']:.3f}, "
                    f"nDCG@5={result['ndcg_at_5']:.3f}"
                )

            all_results[mode] = {
                "recall_at_5": mean_metric(
                    query_results, "recall_at_5"
                ),
                "mrr": mean_metric(query_results, "mrr"),
                "ndcg_at_5": mean_metric(
                    query_results, "ndcg_at_5"
                ),
                "per_query": query_results,
            }

    print("\n========== AGGREGATE RESULTS ==========")

    for mode, metrics in all_results.items():
        print(f"\n{mode}")
        print(f"  Recall@5: {metrics['recall_at_5']:.4f}")
        print(f"  MRR:      {metrics['mrr']:.4f}")
        print(f"  nDCG@5:   {metrics['ndcg_at_5']:.4f}")

    output = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["dataset_version"],
        "workspace_id": workspace_id,
        "top_k": TOP_K,
        "query_count": len(dataset["queries"]),
        "results": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_benchmark()

