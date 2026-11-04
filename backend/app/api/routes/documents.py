from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Document, Workspace
from app.services.document_storage import (
    MAX_FILE_SIZE,
    calculate_sha256,
    save_pdf,
    delete_pdf,
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
