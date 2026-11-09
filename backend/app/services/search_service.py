from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, Workspace
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.rrf_fusion import RRFFusion
from app.vector_store.qdrant_service import QdrantService


class SearchService:
    CANDIDATE_LIMIT = 20

    def __init__(
        self,
        qdrant_service=None,
        bm25_retriever=None,
        rrf_fusion=None,
        reranker=None,
    ):
        self.qdrant_service = qdrant_service or QdrantService()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_fusion = rrf_fusion or RRFFusion(k=60)
        self.reranker = reranker or CrossEncoderReranker()

    def search(
        self,
        db: Session | None = None,
        workspace_id: int = 0,
        query: str = "",
        limit: int = 5,
    ) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if limit <= 0:
            raise ValueError("Search limit must be positive.")

        if db is None:
            from app.db.session import SessionLocal

            with SessionLocal() as session:
                return self.search(
                    db=session,
                    workspace_id=workspace_id,
                    query=query,
                    limit=limit,
                )

        workspace = db.get(Workspace, workspace_id)

        if workspace is None:
            raise ValueError("Workspace not found.")

        candidate_limit = max(self.CANDIDATE_LIMIT, limit * 4)

        dense_results = self.qdrant_service.search(
            query=query,
            workspace_id=workspace_id,
            limit=candidate_limit,
        )

        sparse_results = self.bm25_retriever.search(
            db=db,
            workspace_id=workspace_id,
            query=query,
            limit=candidate_limit,
        )

        fused_results = self.rrf_fusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            limit=candidate_limit,
        )

        if not fused_results:
            return []

        # Rerank the complete fused candidate pool.
        reranked_results = self.reranker.rerank(
            query=query,
            results=fused_results,
            limit=limit,
        )

        document_ids = {
            result["document_id"]
            for result in reranked_results
        }

        documents = db.scalars(
            select(Document).where(
                Document.id.in_(document_ids),
                Document.workspace_id == workspace_id,
            )
        ).all()

        documents_by_id = {
            document.id: document
            for document in documents
        }

        results = []

        for result in reranked_results:
            document = documents_by_id.get(result["document_id"])

            if document is None:
                continue

            results.append({
                "document_id": document.id,
                "filename": document.filename,
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "page_start": result["page_start"],
                "page_end": result["page_end"],
                "score": result["rerank_score"],
            })

        return results
