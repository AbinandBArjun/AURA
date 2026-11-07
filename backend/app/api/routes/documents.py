from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.chunking.section_classifier import SectionClassifier
from app.chunking.text_chunker import TextChunker
from app.models import Document, DocumentPage, DocumentChunk, Workspace
from app.services.document_indexer import (
    DocumentIndexer,
    DocumentIndexingError,
)
from app.services.document_storage import (
    MAX_FILE_SIZE,
    UPLOAD_DIR,
    calculate_sha256,
    save_pdf,
    delete_pdf,
)
from app.services.pdf_extractor import (
    extract_pdf_text,
    PDFExtractionError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Confirm workspace exists.
    workspace = db.get(Workspace, workspace_id)

    if workspace is None:
        raise HTTPException(404, "Workspace not found.")

    # Validate extension.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, "Invalid content type.")

    # Read at most 20 MB + 1 byte to detect oversized files.
    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "PDF exceeds the 20 MB limit.")

    if not content.startswith(b"%PDF-"):
        raise HTTPException(400, "Invalid PDF signature.")

    checksum = calculate_sha256(content)

    # Prevent duplicate documents within the same workspace.
    existing = db.scalar(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.checksum_sha256 == checksum,
        )
    )

    if existing:
        raise HTTPException(409, "This PDF already exists in this workspace.")

    storage_key = None

    try:
        storage_key, saved_checksum = save_pdf(content)

        document = Document(
            workspace_id=workspace_id,
            filename=file.filename,
            content_type="application/pdf",
            size_bytes=len(content),
            checksum_sha256=saved_checksum,
            storage_key=storage_key,
            status="uploaded",
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return {
            "id": document.id,
            "workspace_id": document.workspace_id,
            "filename": document.filename,
            "size_bytes": document.size_bytes,
            "status": document.status,
            "checksum_sha256": document.checksum_sha256,
        }

    except Exception:
        db.rollback()

        if storage_key:
            delete_pdf(storage_key)

        raise

    finally:
        await file.close()


@router.post("/{document_id}/process")
def process_document(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(404, "Document not found.")

    # Resolve and validate the stored file path.
    upload_root = UPLOAD_DIR.resolve()
    file_path = (upload_root / document.storage_key).resolve()

    if not file_path.is_relative_to(upload_root):
        raise HTTPException(400, "Invalid document storage path.")

    if not file_path.is_file():
        raise HTTPException(404, "Stored PDF file not found.")

    try:
        # 1. Extract text page by page.
        result = extract_pdf_text(file_path)

        # 2. Remove old pages and chunks.
        db.execute(
            delete(DocumentPage).where(
                DocumentPage.document_id == document.id
            )
        )

        db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document.id
            )
        )

        # 3. Store extracted pages.
        for page in result["pages"]:
            db.add(
                DocumentPage(
                    document_id=document.id,
                    page_number=page["page_number"],
                    text=page["text"],
                    char_count=len(page["text"]),
                )
            )

        # 4. Generate token-aware chunks.
        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=75,
        )

        classifier = SectionClassifier()

        annotated_segments = classifier.annotate_pages(
            result["pages"]
        )

        chunks = chunker.chunk_pages(annotated_segments)

        # 5. Store chunks in PostgreSQL.
        for chunk in chunks:
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    retrievable=chunk.retrievable,
                )
            )

        # 6. Update extraction status.
        document.extraction_error = None

        if result["total_characters"] == 0:
            document.status = "needs_ocr"
            db.commit()

            return {
                "document_id": document.id,
                "filename": document.filename,
                "status": document.status,
                "page_count": result["page_count"],
                "total_characters": result["total_characters"],
                "chunk_count": len(chunks),
                "indexed_count": 0,
                "message": "No extractable text found. OCR is required.",
            }

        # 7. Persist pages and chunks before vector indexing.
        document.status = "indexing"
        db.commit()

        # 8. Index persisted chunks in Qdrant.
        try:
            indexer = DocumentIndexer()

            indexed_count = indexer.index_document(
                db=db,
                document_id=document.id,
                workspace_id=document.workspace_id,
            )

            expected_indexed_count = sum(
                1 for chunk in chunks if chunk.retrievable
            )

            if indexed_count != expected_indexed_count:
                raise DocumentIndexingError(
                    f"Expected {expected_indexed_count} retrievable vectors, "
                    f"but indexed {indexed_count}."
                )

            document.status = "processed"
            db.commit()

        except Exception as exc:
            db.rollback()

            document = db.get(Document, document_id)
            document.status = "index_failed"
            document.extraction_error = str(exc)
            db.commit()

            raise HTTPException(
                status_code=503,
                detail=f"Document extraction succeeded, "
                       f"but vector indexing failed: {exc}",
            )

        return {
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "page_count": result["page_count"],
            "total_characters": result["total_characters"],
            "chunk_count": len(chunks),
            "indexed_count": indexed_count,
            "message": "PDF extraction, chunking, and vector indexing completed.",
        }

    except PDFExtractionError as exc:
        db.rollback()

        document = db.get(Document, document_id)
        document.status = "failed"
        document.extraction_error = str(exc)
        db.commit()

        raise HTTPException(
            status_code=422,
            detail=f"PDF extraction failed: {exc}",
        )

    except Exception:
        db.rollback()
        raise
