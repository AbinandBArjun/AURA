import re

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk


class BM25Retriever:
    """Workspace-scoped sparse retrieval over stored document chunks."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[\w-]+\b", text.lower())

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

        if not rows:
            return []

        tokenized_corpus = [
            self.tokenize(chunk.text)
            for chunk, _ in rows
        ]

        valid_indices = [
            index
            for index, tokens in enumerate(tokenized_corpus)
            if tokens
        ]

        if not valid_indices:
            return []

        valid_corpus = [
            tokenized_corpus[index]
            for index in valid_indices
        ]

        bm25 = BM25Okapi(valid_corpus)

        # Floor non-positive IDFs so matching terms in small or balanced corpora remain retrievable
        for term in bm25.idf:
            if bm25.idf[term] <= 0:
                bm25.idf[term] = 0.25

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(valid_indices)),
            key=lambda index: (
                float(scores[index]),
                -rows[valid_indices[index]][0].id,
            ),
            reverse=True,
        )

        results = []

        for ranked_index in ranked_indices:
            score = float(scores[ranked_index])

            if score <= 0:
                continue

            chunk, filename = rows[valid_indices[ranked_index]]

            results.append({
                "document_id": chunk.document_id,
                "filename": filename,
                "chunk_id": chunk.id,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "score": score,
            })

            if len(results) >= limit:
                break

        return results
