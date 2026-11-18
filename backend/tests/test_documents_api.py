from sqlalchemy import select

from app.models import Document, User, Workspace
from app.services import document_storage


def create_other_workspace(db_session):
    user = User(
        email="other-user@example.com",
        display_name="Other User",
    )
    db_session.add(user)
    db_session.flush()

    workspace = Workspace(
        owner_id=user.id,
        name="Other Workspace",
    )
    db_session.add(workspace)
    db_session.commit()
    db_session.refresh(workspace)

    return workspace


def test_upload_pdf(
    client,
    current_workspace,
    db_session,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        document_storage,
        "UPLOAD_DIR",
        tmp_path,
    )

    pdf_content = b"%PDF-1.7\nTest PDF content"

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "research.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    result = response.json()

    assert result["workspace_id"] == current_workspace.id
    assert result["filename"] == "research.pdf"
    assert result["size_bytes"] == len(pdf_content)
    assert result["status"] == "uploaded"
    assert result["checksum_sha256"]
    assert result["extraction_error"] is None

    document = db_session.get(Document, result["id"])

    assert document is not None
    assert document.workspace_id == current_workspace.id
    assert (tmp_path / document.storage_key).is_file()


def test_duplicate_upload_returns_conflict(
    client,
    current_workspace,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        document_storage,
        "UPLOAD_DIR",
        tmp_path,
    )

    pdf_content = b"%PDF-1.7\nDuplicate test"

    files = {
        "file": (
            "duplicate.pdf",
            pdf_content,
            "application/pdf",
        )
    }

    first_response = client.post(
        "/api/v1/documents/upload",
        files=files,
    )

    second_response = client.post(
        "/api/v1/documents/upload",
        files=files,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert second_response.json()["detail"] == (
        "This PDF already exists in this workspace."
    )


def test_invalid_pdf_signature_is_rejected(
    client,
    current_workspace,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        document_storage,
        "UPLOAD_DIR",
        tmp_path,
    )

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "invalid.pdf",
                b"This is not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF signature."


def test_document_listing_is_workspace_scoped(
    client,
    current_workspace,
    db_session,
):
    other_workspace = create_other_workspace(db_session)

    current_document = Document(
        workspace_id=current_workspace.id,
        filename="current.pdf",
        content_type="application/pdf",
        size_bytes=100,
        checksum_sha256="current-checksum",
        storage_key="current.pdf",
        status="uploaded",
    )

    other_document = Document(
        workspace_id=other_workspace.id,
        filename="other.pdf",
        content_type="application/pdf",
        size_bytes=100,
        checksum_sha256="other-checksum",
        storage_key="other.pdf",
        status="uploaded",
    )

    db_session.add_all([current_document, other_document])
    db_session.commit()

    response = client.get("/api/v1/documents")

    assert response.status_code == 200

    results = response.json()

    assert len(results) == 1
    assert results[0]["filename"] == "current.pdf"
    assert results[0]["workspace_id"] == current_workspace.id


def test_processing_other_workspace_document_returns_404(
    client,
    current_workspace,
    db_session,
):
    other_workspace = create_other_workspace(db_session)

    document = Document(
        workspace_id=other_workspace.id,
        filename="private.pdf",
        content_type="application/pdf",
        size_bytes=100,
        checksum_sha256="private-checksum",
        storage_key="private.pdf",
        status="uploaded",
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    response = client.post(
        f"/api/v1/documents/{document.id}/process"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

    # The document must remain untouched.
    db_session.refresh(document)
    assert document.status == "uploaded"
