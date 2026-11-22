import json
import math
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from rank_bm25 import BM25Okapi
from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models import Document, DocumentChunk
from app.retrieval.bm25_retriever import BM25Retriever


BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "datasets" / "retrieval_benchmark.json"
OUTPUT_PATH = BASE_DIR / "results" / "bm25_performance_baseline.json"

WARMUP_ROUNDS = 3
MEASURED_ROUNDS = 30
TOP_K = 10

STAGES = [
    "query_tokenization",
    "database_loading",
    "corpus_tokenization",
    "index_construction",
    "scoring_and_ranking",
    "result_materialization",
    "total_profiled",
    "production_search",
]


def percentile(values, percentage):
    ordered = sorted(values)

    if not ordered:
        return 0.0

    index = math.ceil((percentage / 100) * len(ordered)) - 1
    return ordered[max(0, index)]


def summarize(values):
    return {
        "sample_count": len(values),
        "mean_ms": statistics.mean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": percentile(values, 95),
    }


def profile_search(db, retriever, workspace_id, query, limit):
    timings = {stage: 0.0 for stage in STAGES}
    total_start = time.perf_counter()

    start = time.perf_counter()
    query_tokens = retriever.tokenize(query)
    timings["query_tokenization"] = (
        time.perf_counter() - start
    ) * 1000

    if not query_tokens:
        timings["total_profiled"] = (
            time.perf_counter() - total_start
        ) * 1000
        return [], timings

    start = time.perf_counter()

    rows = db.execute(
        select(DocumentChunk, Document.filename)
        .join(
            Document,
            Document.id == DocumentChunk.document_id,
        )
        .where(
            Document.workspace_id == workspace_id,
            DocumentChunk.retrievable.is_(True),
        )
    ).all()

    timings["database_loading"] = (
        time.perf_counter() - start
    ) * 1000

    if not rows:
        timings["total_profiled"] = (
            time.perf_counter() - total_start
        ) * 1000
        return [], timings

    start = time.perf_counter()

    tokenized_corpus = [
        retriever.tokenize(chunk.text)
        for chunk, _ in rows
    ]

    valid_indices = [
        index
        for index, tokens in enumerate(tokenized_corpus)
        if tokens
    ]

    valid_corpus = [
        tokenized_corpus[index]
        for index in valid_indices
    ]

    timings["corpus_tokenization"] = (
        time.perf_counter() - start
    ) * 1000

    if not valid_indices:
        timings["total_profiled"] = (
            time.perf_counter() - total_start
        ) * 1000
        return [], timings

    start = time.perf_counter()

    bm25 = BM25Okapi(valid_corpus)

    for term in bm25.idf:
        if bm25.idf[term] <= 0:
            bm25.idf[term] = 0.25

    timings["index_construction"] = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(valid_indices)),
        key=lambda index: (
            float(scores[index]),
            -rows[valid_indices[index]][0].id,
        ),
        reverse=True,
    )

    timings["scoring_and_ranking"] = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    results = []

    for ranked_index in ranked_indices:
        score = float(scores[ranked_index])

        if score <= 0:
            continue

        chunk, filename = rows[valid_indices[ranked_index]]

        results.append({
            "document_id": chunk.document_id,
            "filename": filename,
            "chunk_id": chunk.id,
            "text": chunk.text,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section": chunk.section,
            "score": score,
        })

        if len(results) >= limit:
            break

    timings["result_materialization"] = (
        time.perf_counter() - start
    ) * 1000

    timings["total_profiled"] = (
        time.perf_counter() - total_start
    ) * 1000

    return results, timings


def run_benchmark():
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    workspace_id = dataset["workspace_id"]
    queries = dataset["queries"]

    retriever = BM25Retriever()

    measurements = {
        stage: []
        for stage in STAGES
    }

    per_query = {}

    print("AURA BM25 Performance Baseline")
    print("=" * 45)
    print("Workspace:", workspace_id)
    print("Queries:", len(queries))
    print("Warm-up rounds:", WARMUP_ROUNDS)
    print("Measured rounds:", MEASURED_ROUNDS)

    with SessionLocal() as db:
        for item in queries:
            query_id = item["query_id"]
            query = item["query"]

            print(f"\nProcessing {query_id}")

            for _ in range(WARMUP_ROUNDS):
                profile_search(
                    db,
                    retriever,
                    workspace_id,
                    query,
                    TOP_K,
                )

                retriever.search(
                    db=db,
                    workspace_id=workspace_id,
                    query=query,
                    limit=TOP_K,
                )

            expected = retriever.search(
                db=db,
                workspace_id=workspace_id,
                query=query,
                limit=TOP_K,
            )

            expected_ids = [
                result["chunk_id"]
                for result in expected
            ]

            query_measurements = {
                stage: []
                for stage in STAGES
            }

            for _ in range(MEASURED_ROUNDS):
                results, timings = profile_search(
                    db,
                    retriever,
                    workspace_id,
                    query,
                    TOP_K,
                )

                actual_ids = [
                    result["chunk_id"]
                    for result in results
                ]

                if actual_ids != expected_ids:
                    raise RuntimeError(
                        f"BM25 result mismatch for {query_id}"
                    )

                production_start = time.perf_counter()

                production_results = retriever.search(
                    db=db,
                    workspace_id=workspace_id,
                    query=query,
                    limit=TOP_K,
                )

                production_latency = (
                    time.perf_counter() - production_start
                ) * 1000

                production_ids = [
                    result["chunk_id"]
                    for result in production_results
                ]

                if production_ids != expected_ids:
                    raise RuntimeError(
                        f"Production search mismatch for {query_id}"
                    )

                timings["production_search"] = production_latency

                for stage in STAGES:
                    value = timings[stage]
                    measurements[stage].append(value)
                    query_measurements[stage].append(value)

            per_query[query_id] = {
                "category": item["category"],
                "measurements": {
                    stage: summarize(values)
                    for stage, values in query_measurements.items()
                },
            }

            print(
                "  Production median:",
                f"{statistics.median(query_measurements['production_search']):.3f} ms",
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

    output = {
        "benchmark": "AURA BM25 Performance Baseline",
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
        "results": {
            stage: summarize(values)
            for stage, values in measurements.items()
        },
        "per_query": per_query,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 45)
    print("AGGREGATE RESULTS")

    for stage, metrics in output["results"].items():
        print(
            f"{stage:25} "
            f"mean={metrics['mean_ms']:.3f} ms  "
            f"median={metrics['median_ms']:.3f} ms  "
            f"p95={metrics['p95_ms']:.3f} ms"
        )

    print("\nResults saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_benchmark()
