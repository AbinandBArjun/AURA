from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def retriever():
    return BM25Retriever()


def test_tokenize(retriever):
    assert retriever.tokenize("AI and Labor Markets") == [
        "ai", "and", "labor", "markets"
    ]


def test_empty_query_raises_error(retriever):
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        retriever.search(
            db=MagicMock(),
            workspace_id=1,
            query=" ",
        )


def test_invalid_limit_raises_error(retriever):
    with pytest.raises(ValueError, match="Search limit must be positive"):
        retriever.search(
            db=MagicMock(),
            workspace_id=1,
            query="artificial intelligence",
            limit=0,
        )


def test_search_returns_relevant_chunks(retriever):
    db = MagicMock()

    rows = [
        (
            SimpleNamespace(
                id=1,
                document_id=1,
                text="AI can improve labor productivity.",
                page_start=2,
                page_end=3,
                section="ai_labor",
            ),
            "paper.pdf",
        ),
        (
            SimpleNamespace(
                id=2,
                document_id=1,
                text="The history of economic policy.",
                page_start=4,
                page_end=5,
                section="body",
            ),
            "paper.pdf",
        ),
    ]

    db.execute.return_value.all.return_value = rows

    results = retriever.search(
        db=db,
        workspace_id=1,
        query="AI labor productivity",
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == 1
    assert results[0]["filename"] == "paper.pdf"
    assert results[0]["section"] == "ai_labor"
    assert results[0]["score"] > 0


def test_no_matching_chunks_returns_empty_list(retriever):
    db = MagicMock()
    db.execute.return_value.all.return_value = []

    results = retriever.search(
        db=db,
        workspace_id=1,
        query="artificial intelligence",
    )

    assert results == []
