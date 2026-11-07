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
        # Load all chunks, including excluded sections, so stale
        # vectors can be removed during re-indexing.
        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()

        # Only eligible chunks should enter semantic retrieval.
        retrievable_chunks = [
            chunk for chunk in chunks if chunk.retrievable
        ]

        try:
            indexed_count = 0

            # Skip upsert when no chunks are eligible.
            if retrievable_chunks:
                indexed_count = self.qdrant_service.upsert_chunks(
                    chunks=retrievable_chunks,
                    workspace_id=workspace_id,
                )

            # Validate the indexing result.
            if indexed_count != len(retrievable_chunks):
                raise DocumentIndexingError(
                    f"Expected {len(retrievable_chunks)} retrievable "
                    f"vectors, but indexed {indexed_count}."
                )

            # Remove vectors for excluded or deleted chunks.
            self.qdrant_service.delete_stale_document_chunks(
                document_id=document_id,
                valid_chunk_ids=[
                    chunk.id for chunk in retrievable_chunks
                ],
            )

            return indexed_count

        except Exception as exc:
            raise DocumentIndexingError(
                f"Failed to index document {document_id}: {exc}"
            ) from exc
