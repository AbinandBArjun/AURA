from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.embeddings.embedding_service import EmbeddingService


class QdrantService:
    COLLECTION_NAME = "aura_document_chunks"
    VECTOR_SIZE = 384

    def __init__(
        self,
        url: str = "http://localhost:6333",
        embedding_service: EmbeddingService | None = None,
    ):
        self.client = QdrantClient(url=url)
        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        # Index fields used for filtered retrieval and cleanup.
        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="workspace_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="retrievable",
            field_schema=models.PayloadSchemaType.BOOL,
        )

        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    def upsert_chunks(
        self,
        chunks: list,
        workspace_id: int,
    ) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]

        vectors = self.embedding_service.embed_texts(texts)

        points = []

        for chunk, vector in zip(chunks, vectors):
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"aura:document:{chunk.document_id}:chunk:{chunk.id}",
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "workspace_id": workspace_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "token_count": chunk.token_count,
                        "section": getattr(chunk, "section", "body"),
                        "retrievable": getattr(chunk, "retrievable", True),
                    },
                )
            )

        self.ensure_collection()

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
            wait=True,
        )

        return len(points)

    def search(
        self,
        query: str,
        workspace_id: int,
        limit: int = 5,
    ) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if limit <= 0:
            raise ValueError("Search limit must be positive.")

        self.ensure_collection()

        query_vector = self.embedding_service.embed_text(query)

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="workspace_id",
                        match=models.MatchValue(value=workspace_id),
                    ),
                    models.FieldCondition(
                        key="retrievable",
                        match=models.MatchValue(value=True),
                    ),
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "score": point.score,
                **point.payload,
            }
            for point in results.points
        ]

    def delete_document_chunks(self, document_id: int) -> None:
        if not self.client.collection_exists(self.COLLECTION_NAME):
            return

        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_stale_document_chunks(
        self,
        document_id: int,
        valid_chunk_ids: list[int],
    ) -> None:
        if not self.client.collection_exists(self.COLLECTION_NAME):
            return

        valid_ids = set(valid_chunk_ids)
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=500,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )

            stale_point_ids = [
                point.id
                for point in points
                if point.payload.get("chunk_id") not in valid_ids
            ]

            if stale_point_ids:
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=models.PointIdsList(
                        points=stale_point_ids,
                    ),
                    wait=True,
                )

            if next_offset is None:
                break

            offset = next_offset
