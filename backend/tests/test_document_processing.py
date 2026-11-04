from unittest.mock import patch

import pymupdf

from sqlalchemy import select, func

from app.chunking.text_chunker import TextChunker
from app.models import Document, DocumentPage, DocumentChunk
from app.api.routes import documents as documents_route


def create_test_pdf(path):
    """Generate a real two-page PDF for integration testing."""
    pdf = pymupdf.open()

    for page_number in range(1, 3):
        page = pdf.new_page()

        text = (
            f"Research findings on page {page_number}. "
            + " ".join(
                ["research evidence methodology analysis"] * 200
            )
        )

        page.insert_textbox(
            pymupdf.Rect(40, 40, 550, 800),
            text,
            fontsize=8,
        )

    pdf.save(path)
    pdf.close()


def test_document_processing_and_reprocessing(
    client,
    db_session,
    test_workspace,
    tmp_path,
    monkeypatch,
):
    # Use temporary storage instead of real uploaded PDFs.
    monkeypatch.setattr(
        documents_route,
        "UPLOAD_DIR",
        tmp_path,
    )

    pdf_path = tmp_path / "integration-test.pdf"
    create_test_pdf(pdf_path)

    document = Document(
        workspace_id=test_workspace.id,
        filename="integration-test.pdf",
        content_type="application/pdf",
        size_bytes=pdf_path.stat().st_size,
        checksum_sha256="test-checksum",
        storage_key=pdf_path.name,
        status="uploaded",
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    # First processing request.
    response = client.post(
        f"/api/v1/documents/{document.id}/process"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "processed"
    assert result["page_count"] == 2
    assert result["chunk_count"] > 0

    # Verify pages and chunks were persisted.
    page_count = db_session.scalar(
        select(func.count())
        .select_from(DocumentPage)
        .where(DocumentPage.document_id == document.id)
    )

    chunk_count = db_session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
    )

    assert page_count == 2
    assert chunk_count == result["chunk_count"]

    # Verify chunk page references are valid.
    chunks = db_session.scalars(
        select(DocumentChunk).where(
            DocumentChunk.document_id == document.id
        )
    ).all()

    assert all(chunk.page_start >= 1 for chunk in chunks)
    assert all(chunk.page_end <= 2 for chunk in chunks)
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)

    # Process the same document again.
    second_response = client.post(
        f"/api/v1/documents/{document.id}/process"
    )

    assert second_response.status_code == 200

    # Reprocessing must not create duplicate pages or chunks.
    new_page_count = db_session.scalar(
        select(func.count())
        .select_from(DocumentPage)
        .where(DocumentPage.document_id == document.id)
    )

    new_chunk_count = db_session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
    )

    assert new_page_count == page_count
    assert new_chunk_count == chunk_count


def test_chunking_failure_preserves_existing_data(
    client,
    db_session,
    test_workspace,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        documents_route,
        "UPLOAD_DIR",
        tmp_path,
    )

    pdf_path = tmp_path / "rollback-test.pdf"
    create_test_pdf(pdf_path)

    document = Document(
        workspace_id=test_workspace.id,
        filename="rollback-test.pdf",
        content_type="application/pdf",
        size_bytes=pdf_path.stat().st_size,
        checksum_sha256="rollback-test-checksum",
        storage_key=pdf_path.name,
        status="uploaded",
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    # First processing succeeds.
    response = client.post(
        f"/api/v1/documents/{document.id}/process"
    )

    assert response.status_code == 200

    original_pages = db_session.scalar(
        select(func.count())
        .select_from(DocumentPage)
        .where(DocumentPage.document_id == document.id)
    )

    original_chunks = db_session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
    )

    assert original_pages == 2
    assert original_chunks > 0

    # Force chunk generation to fail on reprocessing.
    with patch.object(
        TextChunker,
        "chunk_pages",
        side_effect=RuntimeError("Simulated chunking failure"),
    ):
        failed_response = client.post(
            f"/api/v1/documents/{document.id}/process"
        )

    assert failed_response.status_code == 500

    # Previous records must survive the rollback.
    remaining_pages = db_session.scalar(
        select(func.count())
        .select_from(DocumentPage)
        .where(DocumentPage.document_id == document.id)
    )

    remaining_chunks = db_session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
    )

    assert remaining_pages == original_pages
    assert remaining_chunks == original_chunks


