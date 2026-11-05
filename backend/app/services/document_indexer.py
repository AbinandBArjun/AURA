from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import DocumentChunk
from app.vector_store.qdrant_service import QdrantService


class DocumentIndexingError(Exception):
    """Raised when document chunks cannot be indexed."""


class DocumentIndexer:

    def __init__(self, qdrant_service=None):
        self.qdrant_service = qdrant_service or QdrantService()

    def index_document(
        self,
        db: Session,
        document_id: int,
        workspace_id: int,
    ) -> int:
        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()

        if not chunks:
            return 0

        try:
            # 1. Upsert the current vectors.
            indexed_count = self.qdrant_service.upsert_chunks(
                chunks=chunks,
                workspace_id=workspace_id,
            )

            # 2. Validate the indexing result.
            if indexed_count != len(chunks):
                raise DocumentIndexingError(
                    f"Expected {len(chunks)} vectors, "
                    f"but indexed {indexed_count}."
                )

            # 3. Remove obsolete vectors.
            self.qdrant_service.delete_stale_document_chunks(
                document_id=document_id,
                valid_chunk_ids=[chunk.id for chunk in chunks],
            )

            return indexed_count

        except Exception as exc:
            raise DocumentIndexingError(
                f"Failed to index document {document_id}: {exc}"
            ) from exc
