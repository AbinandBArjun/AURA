# AURA — Autonomous Research & Intelligence Platform

[![Core CI](https://github.com/AbinandBArjun/AURA/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AbinandBArjun/AURA/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![React and TypeScript](https://img.shields.io/badge/React%20%2B%20TypeScript-3178C6)

AURA turns a workspace of research PDFs into searchable, citable evidence. It combines section-aware ingestion, dense and lexical retrieval, claim-level verification, and a LangGraph workflow for generating research reports. A React interface supports document processing, search, and multi-turn research conversations.

[Architecture](#architecture) · [Local setup](#local-setup) · [Benchmark results](#retrieval-evaluation) · [Testing and CI](#testing-and-ci)

## What AURA solves

Research PDFs are easy to collect and hard to interrogate. Keyword search can miss paraphrases, semantic search can miss precise terms, and generated answers can hide weak evidence behind fluent prose. AURA brings both retrieval methods together and carries the document, passage, and page range through to the answer. It also measures ranking quality and latency separately so retrieval changes can be judged against a recorded baseline.

| Implemented capability | What it provides |
| --- | --- |
| PDF ingestion | Upload, process, and track PDFs from the Documents page. Persist extracted pages and chunks, then index eligible passages. |
| Knowledge search | Search the active workspace with hybrid retrieval. Inspect returned passage text, source filenames, and page ranges. |
| Evidence-grounded answers | Generate answers from retrieved passages and verify cited claims before returning them in the answer body. |
| Research conversations | Create, reopen, and continue workspace-scoped threads. Follow-up questions can use recent context before retrieval. |
| Research reports | Run a LangGraph research workflow through the API. Receive a cited Markdown report, findings, and claim verification results. |
| Retrieval engineering | Compare dense, hybrid, and reranked retrieval. Check saved reports for aggregate and per-query regressions. |

## Architecture

```mermaid
flowchart LR
    Browser["React workspace"] --> API["FastAPI API"]
    API --> Ingest["PDF ingestion"]
    API --> Search["Hybrid search"]
    API --> RAG["Answers and conversations"]
    API --> Research["LangGraph research workflow"]
    Ingest --> Files["Local PDF storage"]
    Ingest --> SQL[("PostgreSQL")]
    Ingest --> Embed["MiniLM embeddings"]
    Embed --> Qdrant[("Qdrant")]
    Search -->|BM25 chunks and source metadata| SQL
    Search -->|Query embedding and cosine search| Qdrant
    RAG --> Search
    RAG -->|Conversation history| SQL
    RAG -->|Generation and verification| Ollama["Local Ollama"]
    Research --> Search
    Research -->|Plan, analysis, and verification| Ollama
    Research -->|Run metadata and summary| SQL
```

| Responsibility | Stack |
| --- | --- |
| Web interface | React, TypeScript, React Router, Vite |
| API and validation | Python, FastAPI, Pydantic, Uvicorn |
| Structured data | PostgreSQL, SQLAlchemy, psycopg, Alembic |
| PDF processing | PyMuPDF, heading classification, `tiktoken` |
| Retrieval | Sentence Transformers, Qdrant, `rank-bm25`, reciprocal rank fusion, optional cross-encoder |
| Generation and orchestration | Ollama, LangGraph |
| Verification | pytest, GitHub Actions, TypeScript compilation and Vite build |

### Ingestion, chunking, and indexing

1. AURA accepts PDFs up to 20 MiB, checks the file signature, and stores uploads under generated filenames. SHA-256 detects duplicate files within a workspace.
2. PyMuPDF extracts text page by page. The heading classifier annotates segments and excludes references and endnotes from retrieval. Image-only PDFs can reach `needs_ocr`, which indicates that text extraction needs an OCR step outside AURA.
3. The chunker uses `cl100k_base` with a 500-token window and 75-token overlap. It resets across section or retrieval-eligibility changes and records the page range, section, token count, and retrieval flag for each chunk.
4. PostgreSQL stores page text and chunk records. Sentence Transformers batches eligible chunks through `sentence-transformers/all-MiniLM-L6-v2` to produce normalized 384-dimensional vectors. Qdrant stores them with passage text and source metadata. Reprocessing invalidates the workspace BM25 cache and removes stale vectors for the document.

### Retrieval, citations, and research

```mermaid
flowchart LR
    Query["Query"] --> Dense["MiniLM plus Qdrant cosine search"]
    Query --> Sparse["BM25 over eligible workspace chunks"]
    Dense --> Fusion["Reciprocal rank fusion"]
    Sparse --> Fusion
    Fusion --> Top["Top passages for search and RAG"]
    Fusion --> Rerank["Optional cross-encoder"]
    Rerank --> Ranked["Reranked passages"]
    Top --> Draft["Ollama drafts with passage citations"]
    Draft --> Check["Validate citation IDs and verify each claim"]
    Check --> Answer["Supported answer or insufficient evidence"]
```

- **Dense retrieval** searches the Qdrant collection with a query embedding and filters by workspace and retrieval eligibility. **Sparse retrieval** scores lexical matches using BM25 over eligible PostgreSQL chunks from the same workspace.
- **Hybrid retrieval** is the application default. Reciprocal rank fusion combines the two ranked lists by chunk ID using `1 / (60 + rank)` for each contributing list. It works with ranks, so dense and sparse scores do not need to share a scale.
- **Optional reranking** uses `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU to score query–passage pairs from the fused candidate pool. Dense-only and reranked modes are available in the retrieval service and benchmark. The search UI uses the hybrid default.
- **RAG** gives the local Ollama model numbered passages, validates generated citation IDs, and checks each claim against its cited evidence. The answer includes only claims classified as supported. Citations carry filename, document and chunk IDs, and page ranges. If evidence or valid supported claims are insufficient, the API can return an insufficient-evidence response. The verifier is itself model-based.
- **Conversation memory** is stored in PostgreSQL. The service uses up to eight recent messages to rewrite a follow-up into a standalone question, retrieves fresh evidence, and saves the resulting user and assistant turns with citations. A failed rewrite falls back to the original question.
- **Research reports** use a separate backend workflow with planner, researcher, evidence registry, analyst, fact checker, and report generator nodes. The API returns the report and records run status and a findings summary in PostgreSQL. Research, Reports, and Analytics are placeholder routes in the current UI.

### Data and API boundaries

PostgreSQL holds users, workspaces, documents, pages, chunks, conversations, messages, and research runs. A user owns a workspace. A workspace groups documents, conversations, and research runs. Pages and chunks belong to documents, and messages belong to conversations. Qdrant uses a shared cosine collection, `aura_document_chunks`, with indexed workspace, document, and retrieval-eligibility payload fields. Raw uploads live on the application filesystem.

| API under `/api/v1` | Role |
| --- | --- |
| `GET /health`, `GET /workspaces/current` | API liveness and development-workspace bootstrap |
| `GET /documents`, `POST /documents/upload`, `POST /documents/{document_id}/process` | List, upload, and process documents |
| `POST /search`, `POST /rag/answer` | Retrieve passages or generate a cited answer |
| `GET /conversations`, `POST /conversations`, `GET /conversations/{conversation_id}`, `POST /conversations/{conversation_id}/messages` | Manage threads and send messages |
| `POST /research/run` | Execute a research run for a supplied workspace ID and question |

The document, search, RAG, and conversation routes use a development workspace created on demand. Workspace filtering separates data during normal retrieval and lookup. The research endpoint accepts an existing workspace ID directly.

## Retrieval evaluation

The historical [v0.2 dataset](backend/evaluation/datasets/retrieval_benchmark.json) contains 12 queries across six categories and 30 distinct labelled chunk IDs from one paper on AI and the economy. Manually proposed relevance grades range from 0 to 3 and have yet to be independently adjudicated. The runner compares five configurations at a cutoff of five passages and records Precision@5, Recall@5, mean reciprocal rank (MRR), graded nDCG@5, and retrieval latency. These measure passage retrieval, with answer quality outside the benchmark's scope.

| Configuration | Precision@5 | Recall@5 | MRR | nDCG@5 | Mean latency (ms) | Median latency (ms) | p95 latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.5833 | 0.6264 | 0.8611 | 0.7065 | 76.01 | 72.63 | 111.14 |
| Hybrid | 0.6500 | 0.6912 | 1.0000 | 0.8446 | 79.73 | 77.00 | 100.98 |
| Hybrid + rerank 5 candidates | 0.6333 | 0.6847 | 0.9167 | 0.7517 | 282.60 | 285.10 | 303.27 |
| Hybrid + rerank 10 candidates | 0.6667 | 0.7148 | 0.9167 | 0.7615 | 513.30 | 508.44 | 535.99 |
| Hybrid + rerank 20 candidates | 0.7000 | 0.7449 | 0.9167 | 0.7758 | 836.02 | 835.67 | 859.94 |

On this run, hybrid improved all four ranking metrics over dense retrieval at similar mean latency. Reranking 20 candidates yielded the highest recall, while hybrid had the highest MRR and nDCG@5 and lower latency. These are preliminary results from one historical workspace. Its database-generated IDs and existing Qdrant vectors are required to reproduce the live run. [Recorded retrieval results](backend/evaluation/results/retrieval_benchmark_results.json)

The offline regression checker validates report compatibility and checks all four quality metrics by configuration and query. It supports absolute drop tolerances and aggregate minimum scores. The default allows zero degradation. Exit status `0` means pass, `1` indicates a quality regression, and `2` indicates invalid input or incompatible reports. Latency is recorded separately from the quality gate. [Evaluation protocol](backend/evaluation/README.md)

### BM25 performance

The initial sparse-retrieval profile identified corpus tokenization and index construction as the two largest stages. The profiled total had a median of **6.014 ms**, including **2.230 ms** for corpus tokenization and **2.380 ms** for index construction. The uncached application search measured **6.110 ms median / 6.647 ms p95** in that profile. [Tracked BM25 baseline](backend/evaluation/results/bm25_performance_baseline.json)

AURA now caches built indexes in a bounded, thread-safe, process-local cache keyed by workspace. The follow-up benchmark compared forced index rebuilds with reuse of a built index. It used a shared database session, two documents with 37 retrievable chunks, 12 queries, three warm-up rounds and 30 measured rounds per query, and top-10 results. The runner checks result order and score parity against the reference.

| Follow-up BM25 measurement | Mean (ms) | Median (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| Cold BM25 index cache | 11.807 | 11.454 | 14.670 |
| Warm BM25 index cache | 0.305 | 0.285 | 0.565 |

The forced-rebuild median was **about 40.2× the warm median** in this experiment. Cold here means the BM25 index was evicted. The database session was already warm. These timings cover BM25 search on the small recorded corpus. Each API worker keeps its own cache. [Cached BM25 measurements](backend/evaluation/README.md#cached-bm25-performance-benchmark)

## Testing and CI

At committed revision [`66f35cf`](https://github.com/AbinandBArjun/AURA/commit/66f35cf58c407ae06fb29ddd7a5ab769e166b77e), **396 backend tests passed**. All three GitHub Actions jobs were verified, including cold-cache model download and warm-cache restoration. The backend suite covers ingestion, retrieval and metrics, RAG, conversations, research graph and API behavior, and verified model preparation.

| GitHub Actions job | Checks |
| --- | --- |
| Backend core | Python 3.12 compilation and 386 service-independent pytest cases, including benchmark replay and the offline regression checker. Uses SQLite and test doubles. |
| Backend integration | Python 3.12 and 10 pytest cases. Six use a live Qdrant 1.20.0 service pinned by image digest with fake embeddings and isolated test collections. Four exercise the real MiniLM model offline. |
| Frontend build | Node.js 24, `npm ci`, TypeScript compilation, and Vite build. |

The [workflow](.github/workflows/ci.yml) runs on Ubuntu 24.04 for pull requests to `main`, pushes to `main`, and manual runs. It runs the jobs independently, cancels superseded runs, and uses timeouts, SHA-pinned actions, read-only repository permissions, and checkout without persisted credentials. pip and npm downloads are cached using their dependency files. The integration job waits for Qdrant readiness and prints container logs on failure. CI requires no repository secrets, PostgreSQL, or Ollama. Live retrieval benchmarking and continuous deployment are outside the workflow.

### Model integrity and offline inference

The integration [model manifest](backend/tests/embedding_model_manifest.json) pins MiniLM to upstream revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and specifies SHA-256 hashes for ten inference files, including the Safetensors weights. A cold run downloads only the required files into a temporary directory and publishes the model directory after validation. A warm run validates the restored files locally. The fixture verifies the prepared files again before loading the model for offline inference with remote model code disabled.

The model cache key includes the runner OS, architecture, preparation script, and manifest. A bad cache fails with a diagnostic and leaves the files in place. GitHub Actions cache entries are immutable. Remove the affected entry and rerun to repopulate it. Local application inference uses the model IDs in the application code. The verified pin and offline fixture apply to integration tests.

## Security controls and current limits

- The upload route checks file size and PDF signature, uses generated storage keys, and checks for duplicate content within the workspace. CORS permits the local frontend origin. Workspace filters apply to the relevant data queries and retrieval paths.
- RAG checks citation integrity and filters unsupported claims. The checks reduce unsupported output within the implemented workflow. The verifier can still make errors.
- The application is designed for local development. The development workspace is bootstrapped without authentication, and the research API takes a workspace ID. Authenticated user management, production tenant authorization, and a security audit remain future work. Keep application and service ports limited to a trusted development machine or network.
- The backend Dockerfile provides a basic API image. Compose starts PostgreSQL and Qdrant. The setup below runs the backend, frontend, and Ollama separately. The repository has no automated cloud deployment.

## Local setup

Install Git, Python 3.12, Node.js 24 with npm, and Docker with Compose. Install Ollama for answers, multi-turn RAG, and research reports. Ingestion and search use the embedding model and can run without Ollama. The first dependency and model downloads need network access unless the assets are already cached. Use a text-based PDF for the first run.

### 1. Clone and configure

```bash
git clone https://github.com/AbinandBArjun/AURA.git
cd AURA
cp .env.example .env
```

In PowerShell, use `Copy-Item .env.example .env`. Edit the root `.env` with your local values. Replace `YOUR_LOCAL_DB_PASSWORD` in both lines with the same local password. URL-encode the password in `DATABASE_URL` when it contains reserved URL characters. The example file's PostgreSQL port needs to match the host port in Compose, **5433**.

```dotenv
ENVIRONMENT=development
POSTGRES_PASSWORD=YOUR_LOCAL_DB_PASSWORD
DATABASE_URL=postgresql+psycopg://aura:YOUR_LOCAL_DB_PASSWORD@localhost:5433/aura
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

| Setting | Use |
| --- | --- |
| `POSTGRES_PASSWORD`, `DATABASE_URL` | Compose initializes the local PostgreSQL volume with the password. The backend and Alembic use the connection URL. |
| `ENVIRONMENT` | Set to `development` for the local workspace bootstrap. |
| `DEVELOPMENT_USER_EMAIL`, `DEVELOPMENT_USER_NAME`, `DEVELOPMENT_WORKSPACE_NAME` | Optional overrides for the locally bootstrapped user and workspace. |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Local Ollama endpoint and model name. Pull the model you select. |
| `QDRANT_URL`, `EMBEDDING_MODEL` | Declared in settings. The indexing and retrieval code uses localhost Qdrant and the MiniLM model constant directly. |
| `VITE_API_BASE_URL` | Optional frontend override in the shell or `frontend/.env.local`. Defaults to `http://localhost:8000/api/v1`. Browser-visible values must contain no secrets. |

The optional provider-key fields in `.env.example` can stay empty for this local Ollama workflow.

### 2. Start the data services and optional Ollama model

From the repository root:

```bash
docker compose up -d postgres qdrant
docker compose ps
```

PostgreSQL is available on host port `5433` and Qdrant HTTP on `6333`. After starting Ollama through its desktop app or with `ollama serve` in a separate terminal, pull the configured model:

```bash
ollama pull qwen2.5:3b
```

If this model exceeds your machine’s resources, choose a smaller compatible model and set `OLLAMA_MODEL` to its name.

### 3. Start the API

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

In PowerShell, replace the activation command with `.\.venv\Scripts\Activate.ps1`. The API serves [Swagger UI](http://localhost:8000/docs) and a [liveness endpoint](http://localhost:8000/api/v1/health). Liveness only confirms the API process responds.

### 4. Start the web application

From the repository root in a new terminal:

```bash
cd frontend
npm ci
npm run dev -- --host localhost
```

Open [http://localhost:5174](http://localhost:5174). In Documents, upload and process a text-based PDF. In Search, query something covered by the document and inspect the returned passage and page range. With Ollama running, ask a cited question or open Conversations and ask a follow-up. To try the research graph, use `POST /api/v1/research/run` in Swagger UI with the ID returned by `GET /api/v1/workspaces/current` and a question about the indexed material.

## Run checks and benchmarks

Run these commands from `backend` in the Python 3.12 environment. Set the offline flags for core checks. A cold tokenizer cache may still fetch the public `tiktoken` vocabulary.

```bash
export DATABASE_URL=sqlite:// ENVIRONMENT=development
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m compileall -q app evaluation migrations tests
python -m pytest -q --strict-markers -m "not live_qdrant and not real_embedding_model"
```

In PowerShell, use `$env:NAME = 'value'` for the corresponding environment variables. For integration tests, use a disposable Qdrant instance at `localhost:6333`. One local option is `docker compose up -d qdrant` from the root. Check readiness and prepare the pinned model from `backend`:

```bash
curl --fail --show-error http://localhost:6333/readyz
export DATABASE_URL=sqlite:// ENVIRONMENT=development
export AURA_TEST_EMBEDDING_MODEL="$(mktemp -d)/minilm"
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python tests/prepare_embedding_model.py "$AURA_TEST_EMBEDDING_MODEL"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest -q --strict-markers -m "live_qdrant or real_embedding_model"
# Run both test selections with the same service and model available.
python -m pytest -q --strict-markers
```

In PowerShell, set `AURA_TEST_EMBEDDING_MODEL` to a new local directory, run `curl.exe` for the readiness check, and set the environment flags before running each Python command. From `frontend`, run `npm ci` and `npm run build` for the production build used by CI.

Use a separate shell with the historical benchmark's PostgreSQL and Qdrant configuration for live evaluation. The original workspace, labelled chunks, vectors, and embedding and reranking model assets are required. Retrieval evaluation runs without Ollama. A freshly migrated database does not contain that corpus.

```bash
# From backend, with the matching benchmark data and services available.
python -m evaluation.validate_dataset
python -m evaluation.run_retrieval_benchmark --output evaluation/results/retrieval_benchmark_current.json

# Offline JSON quality gate, after producing a candidate report.
python -m evaluation.check_retrieval_regression evaluation/results/retrieval_benchmark_current.json

# Isolated BM25 profiling, using PostgreSQL.
python -m evaluation.run_bm25_optimized_performance
```

Always supply `--output` for a new retrieval report so the tracked baseline is preserved. The original profiler, `python -m evaluation.run_bm25_performance`, writes to the tracked BM25 baseline path and belongs in a disposable copy when reproducing that profile. [Full evaluation commands and output formats](backend/evaluation/README.md)

## Repository guide

```text
.github/workflows/       GitHub Actions verification
backend/
  app/
    agents/              LangGraph research workflow
    api/                 FastAPI routes and dependencies
    chunking/            Heading classification and token windows
    embeddings/          Sentence Transformers inference
    models/              SQLAlchemy models
    retrieval/           BM25, index cache, RRF, cross-encoder
    services/            Ingestion, search, RAG, conversations, research
    vector_store/        Qdrant collection and retrieval
  evaluation/            Datasets, metrics, scripts, historical reports
  migrations/            Alembic schema history
  tests/                 Unit, API, replay, and integration tests
frontend/src/            React pages, layout, and API client
docker-compose.yml       Local PostgreSQL and Qdrant
.env.example             Local configuration template
```

For further detail, see the [evaluation guide](backend/evaluation/README.md), [API route implementations](backend/app/api/routes), [Alembic migrations](backend/migrations/versions), and [CI workflow](.github/workflows/ci.yml).

## Roadmap

| Status | Scope |
| --- | --- |
| Completed | Milestones 1–16 and checkpoints 17.1–17.3, covering the application, retrieval evaluation, regression checks, and three-job CI. |
| In progress | Checkpoint 17.4.2 is reviewed local work, pending commit and publication. It introduces a fictional portable corpus with eight documents, 40 logical passages, and 36 draft queries. Relevance judgments and portable benchmark scores are pending. Local verification of the pending changes recorded 429 passed tests and two environment-specific symbolic-link skips. |
| Future work | Review relevance labels, add isolated benchmark seeding, and reproduce retrieval evaluation from a fresh workspace. Owner approval governs publication of the portable corpus. |
