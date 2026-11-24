# AURA — Autonomous Research & Intelligence Platform

AURA is an agentic research system that ingests documents, retrieves evidence, verifies claims and generates cited research reports.

## Current starter scope
- FastAPI application and health endpoint
- React + TypeScript + Vite frontend shell
- PostgreSQL and Qdrant local services
- Initial SQLAlchemy models: users, workspaces, documents and research runs

## Requirements
- Python 3.11 or 3.12
- Node.js 20+
- Docker Desktop / Docker Engine with Compose
- Ollama installed locally (optional in the first setup step)

## Start infrastructure
```bash
docker compose up -d
cp .env.example .env
```

## Start backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs: http://localhost:8000/docs
Health: http://localhost:8000/api/v1/health

## Start frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173

## Continuous integration

The [Core CI workflow](.github/workflows/ci.yml) runs on pull requests to
`main`, pushes to `main`, and manual dispatch. The backend core, backend integration,
and frontend jobs run independently on Ubuntu. The backend jobs
use Python 3.12, and the frontend uses Node.js 24 LTS.

Run the backend core checks locally from `backend` with Python 3.12. The core job
sets a SQLite database URL, keeps the development workspace mode used by
the tests, and disables Hugging Face model downloads.

```bash
export DATABASE_URL=sqlite:// ENVIRONMENT=development
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pip install -r requirements.txt
python -m compileall -q app evaluation migrations tests
python -m pytest -q --strict-markers -m "not live_qdrant and not real_embedding_model"
```

The selection includes the retrieval regression checker and benchmark
replay tests. It deselects these test modules by their pytest markers.

| Marker | Excluded tests | Requirement |
|---|---|---|
| `live_qdrant` | `tests/test_qdrant_service.py` (6 tests) | Running Qdrant HTTP server at `localhost:6333`. The tests supply fake embeddings. |
| `real_embedding_model` | `tests/test_embeddings.py` (4 tests) | Real `sentence-transformers/all-MiniLM-L6-v2` weights in the local cache or available from Hugging Face. |

The core job starts no service containers. Chunking tests may fetch the
public `tiktoken` vocabulary on a cold runner.

Run the frontend production build from `frontend` with Node.js 24.

```bash
npm ci
npm run build
```

The workflow caches pip and npm downloads using `backend/requirements.txt`
and `frontend/package-lock.json` respectively.

### Backend integration tests

The integration job runs the complementary marker selection. Together the
core and integration jobs cover the full backend suite. The integration
job starts a fresh Qdrant 1.20.0 container pinned by image digest and waits
for `http://localhost:6333/readyz`. If readiness or a test fails, the job
prints the available Qdrant container logs. Each Qdrant test uses fake
embeddings and creates a uniquely named collection. Cleanup deletes only
a collection the fixture successfully created. Use a disposable Qdrant
instance for local integration testing.

The four embedding tests load `sentence-transformers/all-MiniLM-L6-v2`
offline. The checked-in model manifest pins the upstream revision and
SHA-256 for every required inference file. On a cache miss, preparation
downloads only these files into a temporary directory and publishes the
model after validating it. On a cache hit, preparation validates the
restored files without contacting Hugging Face. The embedding fixture
validates again before loading the model.

The cache key includes the runner OS, architecture, preparation script,
and integrity manifest. If the restored directory is invalid, the job
fails with the offending path and leaves the files in place. GitHub
Actions cache entries are immutable. Delete the affected cache entry and
rerun the workflow to populate it again. Locally, prepare a new directory
when a previous model directory fails validation.

To run the integration tests locally, start a compatible Qdrant server on
port 6333 and install the backend requirements with Python 3.12. The
repository's Compose configuration can start Qdrant by itself with
`docker compose up -d qdrant` from the repository root. Then run from
`backend`:

```bash
curl --fail --show-error http://localhost:6333/readyz
export DATABASE_URL=sqlite:// ENVIRONMENT=development
export AURA_TEST_EMBEDDING_MODEL="$(mktemp -d)/minilm"
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python tests/prepare_embedding_model.py "$AURA_TEST_EMBEDDING_MODEL"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q --strict-markers -m "live_qdrant or real_embedding_model"
```

The integration job uses no repository secrets, PostgreSQL, or Ollama. A cold
model cache requires public Hugging Face access. Live retrieval benchmarking
and deployment are outside this workflow.

## Optional local LLM
Install Ollama from https://ollama.com and then:
```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```
Model availability and hardware requirements vary. If this model is too large for your machine, choose a smaller compatible model and update OLLAMA_MODEL.

## Important
The database models are an initial schema foundation, not a production-ready authentication system. Add Alembic migrations before evolving schemas. Add authentication and workspace-level authorization before exposing user data beyond local development.
