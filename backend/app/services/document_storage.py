import hashlib
from pathlib import Path
from uuid import uuid4

UPLOAD_DIR = (
    Path(__file__).resolve().parents[1]
    / "storage"
    / "uploads"
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def calculate_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_pdf(content: bytes) -> tuple[str, str]:
    """
    Save a validated PDF using a generated filename.
    Returns (storage_key, checksum).
    """
    if not content:
        raise ValueError("The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("PDF exceeds the 20 MB limit.")

    if not content.startswith(b"%PDF-"):
        raise ValueError("File does not have a valid PDF signature.")

    checksum = calculate_sha256(content)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    storage_key = f"{uuid4().hex}.pdf"
    destination = UPLOAD_DIR / storage_key

    destination.write_bytes(content)

    return storage_key, checksum


def delete_pdf(storage_key: str) -> None:
    """Delete a stored PDF if it exists."""
    # Only allow generated UUID-based filenames.
    if Path(storage_key).name != storage_key:
        raise ValueError("Invalid storage key.")

    path = UPLOAD_DIR / storage_key

    if path.exists():
        path.unlink()
