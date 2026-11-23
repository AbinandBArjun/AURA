import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.retrieval.bm25_index_cache import (
    BM25IndexCache,
    bm25_index_cache,
)


@dataclass(frozen=True)
class BM25Chunk:
    document_id: int
    filename: str
    chunk_id: int
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None


@dataclass(frozen=True)
class BM25IndexSnapshot:
    chunks: tuple[BM25Chunk, ...]
    bm25: Any | None


class BM25Retriever:
    """Workspace-scoped sparse retrieval with cached BM25 indexes."""

    def __init__(self, index_cache: BM25IndexCache | None = None):
        self.index_cache = (
            index_cache if index_cache is not None else bm25_index_cache
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[\w-]+\b", text.lower())

    def _build_index(
        self,
        db: Session,
        workspace_id: int,
    ) -> BM25IndexSnapshot:
        rows = db.execute(
            select(DocumentChunk, Document.filename)
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .where(
                Document.workspace_id == workspace_id,
                DocumentChunk.retrievable.is_(True),
            )
        ).all()

        chunks = []
        tokenized_corpus = []

        for chunk, filename in rows:
            tokens = self.tokenize(chunk.text)

            if not tokens:
                continue

            chunks.append(
                BM25Chunk(
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_id=chunk.id,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                )
            )
            tokenized_corpus.append(tokens)

        if not tokenized_corpus:
            return BM25IndexSnapshot(
                chunks=(),
                bm25=None,
            )

        bm25 = BM25Okapi(tokenized_corpus)

        # Preserve the existing IDF floor.
        for term in bm25.idf:
            if bm25.idf[term] <= 0:
                bm25.idf[term] = 0.25

        return BM25IndexSnapshot(
            chunks=tuple(chunks),
            bm25=bm25,
        )

    def search(
        self,
        db: Session,
        workspace_id: int,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if limit <= 0:
            raise ValueError("Search limit must be positive.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        snapshot = self.index_cache.get_or_create(
            workspace_id,
            lambda: self._build_index(db, workspace_id),
        )

        if snapshot.bm25 is None:
            return []

        scores = snapshot.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(snapshot.chunks)),
            key=lambda index: (
                float(scores[index]),
                -snapshot.chunks[index].chunk_id,
            ),
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = snapshot.chunks[index]

            results.append({
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "score": score,
            })

            if len(results) >= limit:
                break

        return results
