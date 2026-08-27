import pytest

from app.domain.errors import DocumentTooLargeError, InvalidDocumentError
from app.infrastructure.documents.validation import PdfUploadValidator


def test_validator_sanitizes_path_and_unsafe_filename_characters() -> None:
    validator = PdfUploadValidator(max_upload_size_mb=1)

    filename = validator.validate("../policies/warranty<script>.PDF", b"%PDF-1.7")

    assert filename == "warranty_script.pdf"


@pytest.mark.parametrize("filename", ["document.txt", "document.pdf.exe", ""])
def test_validator_rejects_non_pdf_filename(filename: str) -> None:
    validator = PdfUploadValidator(max_upload_size_mb=1)

    with pytest.raises(InvalidDocumentError, match=".pdf extension"):
        validator.validate(filename, b"%PDF-1.7")


def test_validator_rejects_spoofed_pdf() -> None:
    validator = PdfUploadValidator(max_upload_size_mb=1)

    with pytest.raises(InvalidDocumentError, match="valid PDF"):
        validator.validate("document.pdf", b"plain text")


def test_validator_enforces_upload_size_limit() -> None:
    validator = PdfUploadValidator(max_upload_size_mb=1)
    oversized_content = b"%PDF-" + bytes(1024 * 1024)

    with pytest.raises(DocumentTooLargeError, match="1 MB"):
        validator.validate("document.pdf", oversized_content)
