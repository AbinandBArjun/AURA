import pymupdf
import pytest

from app.services.pdf_extractor import (
    PDFExtractionError,
    extract_pdf_text,
)


def test_extract_pdf_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "AURA research platform test.")
    document.save(pdf_path)
    document.close()

    result = extract_pdf_text(pdf_path)

    assert result["page_count"] == 1
    assert result["total_characters"] > 0
    assert "AURA research platform test." in result["pages"][0]["text"]
    assert result["pages"][0]["page_number"] == 1


def test_missing_pdf_raises_error(tmp_path):
    with pytest.raises(PDFExtractionError):
        extract_pdf_text(tmp_path / "missing.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PDF")

    with pytest.raises(PDFExtractionError):
        extract_pdf_text(file_path)
