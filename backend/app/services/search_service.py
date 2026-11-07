from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, Workspace
from app.vector_store.qdrant_service import QdrantService


class SearchService:

    def __init__(self, qdrant_service=None):
        self.qdrant_service = qdrant_service or QdrantService()

    def search(
        self,
        db: Session | None = None,
        workspace_id: int = 0,
        query: str = "",
        limit: int = 5,
    ) -> list[dict]:
        if db is None:
            from app.db.session import SessionLocal

            with SessionLocal() as session:
                return self.search(
                    db=session,
                    workspace_id=workspace_id,
                    query=query,
                    limit=limit,
                )

        # Confirm that the workspace exists.
        workspace = db.get(Workspace, workspace_id)

        if workspace is None:
            raise ValueError("Workspace not found.")

        # Retrieve semantically similar chunks from Qdrant.
        vector_results = self.qdrant_service.search(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
        )

        if not vector_results:
            return []

        # Retrieve document metadata from PostgreSQL.
        document_ids = {
            result["document_id"] for result in vector_results
        }

        documents = db.scalars(
            select(Document).where(
                Document.id.in_(document_ids),
                Document.workspace_id == workspace_id,
            )
        ).all()

        documents_by_id = {
            document.id: document for document in documents
        }

        # Combine vector results with document metadata.
        results = []

        for result in vector_results:
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
                "score": result["score"],
            })

        return results

