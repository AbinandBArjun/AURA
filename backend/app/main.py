from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.search import router as search_router

app = FastAPI(
    title="AURA API",
    description="Autonomous Research & Intelligence Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")


@app.get("/")
def root():
    return {"name": "AURA API", "version": "0.1.0", "docs": "/docs"}
