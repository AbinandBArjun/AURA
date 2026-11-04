import pytest

from app.chunking.text_chunker import TextChunker


def test_chunker_splits_long_text():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)

    pages = [
        {
            "page_number": 1,
            "text": " ".join(["research"] * 200),
        }
    ]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 50 for chunk in chunks)
    assert all(chunk.page_start == 1 for chunk in chunks)


def test_chunker_preserves_page_references():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)

    pages = [
        {"page_number": 1, "text": " ".join(["first"] * 30)},
        {"page_number": 2, "text": " ".join(["second"] * 30)},
    ]

    chunks = chunker.chunk_pages(pages)

    assert chunks
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    assert any(chunk.page_end == 2 for chunk in chunks)


def test_chunker_skips_empty_pages():
    chunker = TextChunker()

    chunks = chunker.chunk_pages([
        {"page_number": 1, "text": "   "},
        {"page_number": 2, "text": "Useful research content."},
    ])

    assert len(chunks) == 1
    assert chunks[0].page_start == 2


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_zero_overlap_does_not_repeat_chunks():
    chunker = TextChunker(chunk_size=20, chunk_overlap=0)

    pages = [
        {"page_number": 1, "text": " ".join(["research"] * 100)}
    ]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 20 for chunk in chunks)


