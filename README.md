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

## Optional local LLM
Install Ollama from https://ollama.com and then:
```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```
Model availability and hardware requirements vary. If this model is too large for your machine, choose a smaller compatible model and update OLLAMA_MODEL.

## Important
The database models are an initial schema foundation, not a production-ready authentication system. Add Alembic migrations before evolving schemas. Add authentication and workspace-level authorization before exposing user data beyond local development.
