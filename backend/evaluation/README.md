# AURA Retrieval Evaluation

This directory contains AURA's retrieval evaluation framework, including
the benchmark dataset, dataset validator, information retrieval metrics,
benchmark runner, and generated results.

## 1. Evaluation Overview

AURA compares three retrieval strategies:

| Strategy | Description |
|---|---|
| Dense | Vector similarity retrieval using Qdrant |
| Hybrid | Dense + BM25 retrieval combined using Reciprocal Rank Fusion (RRF) |
| Hybrid + Reranking | Hybrid retrieval followed by cross-encoder reranking |

The production default is **hybrid retrieval**. Cross-encoder reranking
remains available as an optional retrieval mode.

## 2. Evaluation Metrics

- Precision@5: Fraction of the top five retrieved passages that are relevant.
- Recall@5: Fraction of all positively labelled relevant passages retrieved in the top five.
- MRR: Mean Reciprocal Rank of the first relevant result.
- nDCG@5: Ranking quality using graded relevance labels.
- Mean latency: Average retrieval latency in milliseconds.
- Median latency: Median retrieval latency in milliseconds.
- p95 latency: 95th-percentile retrieval latency in milliseconds.

Relevance grades range from 0 to 3:

| Grade | Meaning |
|---|---|
| 0 | Not relevant |
| 1 | Marginally relevant |
| 2 | Relevant |
| 3 | Highly relevant |

## 3. Benchmark Dataset

Dataset: `AURA Retrieval Benchmark`

Version: `0.2`

Location: `evaluation/datasets/retrieval_benchmark.json`

Current dataset characteristics:

- 12 evaluation queries.
- 6 query categories.
- 30 unique labelled passage identifiers.
- One research document about AI and the economy.
- Manually proposed graded relevance judgements.
- Workspace ID 1.

The relevance labels have not been independently adjudicated.
Results should therefore be interpreted as preliminary.

## 4. Benchmark Results

Results below are from the latest benchmark execution using version 0.2
of the dataset and a retrieval cutoff of five passages.

| Configuration | Precision@5 | Recall@5 | MRR | nDCG@5 | Mean latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.5833 | 0.6264 | 0.8611 | 0.7065 | 76.01 ms |
| Hybrid | 0.6500 | 0.6912 | 1.0000 | 0.8446 | 79.73 ms |
| Hybrid + Rerank (5) | 0.6333 | 0.6847 | 0.9167 | 0.7517 | 282.60 ms |
| Hybrid + Rerank (10) | 0.6667 | 0.7148 | 0.9167 | 0.7615 | 513.30 ms |
| Hybrid + Rerank (20) | 0.7000 | 0.7449 | 0.9167 | 0.7758 | 836.02 ms |

### Latency Distribution

| Configuration | Median latency | p95 latency |
|---|---:|---:|
| Dense | 72.63 ms | 111.14 ms |
| Hybrid | 77.00 ms | 100.98 ms |
| Hybrid + Rerank (5) | 285.10 ms | 303.27 ms |
| Hybrid + Rerank (10) | 508.44 ms | 535.99 ms |
| Hybrid + Rerank (20) | 835.67 ms | 859.94 ms |

### Interpretation

**Hybrid retrieval is the current production default.**

Compared with dense retrieval, hybrid retrieval improves all four
ranking metrics:

- Precision@5: 0.5833 to 0.6500.
- Recall@5: 0.6264 to 0.6912.
- MRR: 0.8611 to 1.0000.
- nDCG@5: 0.7065 to 0.8446.

The latest run measured comparable mean latency for dense and hybrid
retrieval: 76.01 ms and 79.73 ms respectively.

Increasing the reranking candidate pool improves precision and recall
on this dataset, but introduces substantial additional latency.

Reranking 20 candidates achieves Recall@5 of 0.7449, compared with
0.6912 for hybrid retrieval. However, mean latency rises to 836.02 ms.
Its MRR and nDCG@5 also remain below the hybrid configuration.

Reranking is therefore retained as an optional capability rather than
the production default.

## 5. Reproducibility

Run benchmark commands from the `backend` directory.

### Prerequisites

- Python environment with `backend/requirements.txt` installed.
- PostgreSQL running through Docker Compose.
- Qdrant running through Docker Compose.
- Project-root `.env` configured with the appropriate service URLs.
- Benchmark workspace and labelled document chunks already ingested.
- Embedding model available.
- Cross-encoder model available for reranking configurations.

Start infrastructure from the project root:

```powershell
docker compose up -d postgres qdrant
```

### Validate Dataset

Run the validator to confirm all labelled chunks are present and retrievable:

```powershell
python -m evaluation.validate_dataset
```

### Run Benchmark

Execute the evaluation across dense, hybrid, and reranking configurations:

```powershell
python -m evaluation.run_retrieval_benchmark
```

### Benchmark Output

Results are written to:

`evaluation/results/retrieval_benchmark_results.json`

### Isolated BM25 Performance Benchmark

The isolated BM25 benchmark profiles the individual stages of sparse
retrieval, separately from the end-to-end retrieval benchmark.

Run it from the `backend` directory:

```powershell
python -m evaluation.run_bm25_performance
```

Results are written to:

`evaluation/results/bm25_performance_baseline.json`


#### Recorded Baseline

The baseline was measured with the following configuration:

- Dataset: AURA Retrieval Benchmark, version 0.2.
- Workspace: 1.
- Corpus: 2 documents, 50 total chunks, 37 retrievable chunks.
- BM25 implementation: `rank-bm25` 0.2.2.
- Queries: 12.
- Warm-up rounds: 3 per query.
- Measured rounds: 30 per query.
- Result limit: top 10.

Aggregate timings:

| Stage | Median (ms) | P95 (ms) |
|---|---:|---:|
| Query tokenization | 0.006 | 0.008 |
| Database loading | 1.200 | 1.578 |
| Corpus tokenization | 2.230 | 2.431 |
| Index construction | 2.380 | 2.591 |
| Scoring and ranking | 0.165 | 0.213 |
| Result materialization | 0.017 | 0.018 |
| Total profiled | 6.014 | 6.577 |
| Production search | 6.110 | 6.647 |

Each aggregate stage contains 360 measurements. The benchmark also
records per-query measurements and verifies that the profiled and
production implementations return identical ordered chunk IDs.

Corpus tokenization and index construction account for approximately
76.7% of the total profiled median. They are the largest measured
components of this baseline.

**Measurement limitation:** the benchmark uses a shared SQLAlchemy
session across queries and performs warm-up rounds before measurement.
These results represent a warm-session baseline for the recorded
37-retrievable-chunk corpus. They do not establish cold-start latency
or performance at larger corpus sizes. The measurements should be
repeated against representative larger corpora before making
scaling or caching decisions.

### Cached BM25 Performance Benchmark

The optimized benchmark measures BM25 retrieval with workspace-scoped index caching.

Run from the backend directory:

    python -m evaluation.run_bm25_optimized_performance

Results: evaluation/results/bm25_performance_optimized.json

| Measurement | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| Cold-cache search | 11.807 | 11.454 | 14.670 |
| Warm-cache search | 0.305 | 0.285 | 0.565 |

The measured warm-cache median was approximately 40.17 times faster than the cold-cache median.

Retrieval-quality metrics remained exactly equal across all five retrieval modes.

The cache is bounded, thread-safe, workspace-scoped, and process-local. Separate worker processes maintain independent caches. Larger production workloads require further benchmarking.
## 6. Directory Structure

- `datasets/`: Benchmark dataset (`retrieval_benchmark.json`).
- `metrics/`: Information retrieval metrics (`retrieval_metrics.py`).
- `results/`: Benchmark execution output (`retrieval_benchmark_results.json`, `bm25_performance_baseline.json`).
- `run_retrieval_benchmark.py`: Benchmark runner script.
- `run_bm25_performance.py`: Isolated BM25 performance profiler.
- `run_bm25_optimized_performance.py`: Cached BM25 performance benchmark.
- `validate_dataset.py`: Benchmark dataset validator.