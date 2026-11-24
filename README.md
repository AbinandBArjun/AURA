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
`main`, pushes to `main`, and manual dispatch. The independent backend and
frontend jobs use Ubuntu runners with Python 3.12 and Node.js 24 LTS.

Run the backend checks locally from `backend` with Python 3.12. The workflow
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

Core CI starts no service containers and uses no application secrets.
Chunking tests may fetch the public `tiktoken` vocabulary on a cold
runner. The live retrieval benchmark and deployment are outside this
workflow.

Run the frontend production build from `frontend` with Node.js 24.

```bash
npm ci
npm run build
```

The workflow caches pip and npm downloads using `backend/requirements.txt`
and `frontend/package-lock.json` respectively.

## Optional local LLM
Install Ollama from https://ollama.com and then:
```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```
Model availability and hardware requirements vary. If this model is too large for your machine, choose a smaller compatible model and update OLLAMA_MODEL.

## Important
The database models are an initial schema foundation, not a production-ready authentication system. Add Alembic migrations before evolving schemas. Add authentication and workspace-level authorization before exposing user data beyond local development.
