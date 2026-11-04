from pathlib import Path

import pymupdf


class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""


def extract_pdf_text(file_path: str | Path) -> dict:
    """
    Extract text from a PDF while preserving page-level information.

    Returns:
        {
            "page_count": int,
            "total_characters": int,
            "pages": [
                {"page_number": int, "text": str}
            ]
        }
    """
    path = Path(file_path)

    if not path.exists():
        raise PDFExtractionError("PDF file does not exist.")

    if not path.is_file():
        raise PDFExtractionError("Provided path is not a file.")

    if path.suffix.lower() != ".pdf":
        raise PDFExtractionError("Only PDF files are supported.")

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise PDFExtractionError(
                    "Password-protected PDFs are not supported."
                )

            pages = []

            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()

                pages.append({
                    "page_number": page_index + 1,
                    "text": text,
                })

            total_characters = sum(
                len(page["text"]) for page in pages
            )

            return {
                "page_count": len(pages),
                "total_characters": total_characters,
                "pages": pages,
            }

    except PDFExtractionError:
        raise

    except Exception as exc:
        raise PDFExtractionError(
            f"Could not extract PDF text: {exc}"
        ) from exc
