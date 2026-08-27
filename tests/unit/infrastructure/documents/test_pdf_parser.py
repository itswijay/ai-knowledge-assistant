from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.domain.errors import DocumentParsingError, InvalidDocumentError
from app.infrastructure.documents.pdf_parser import PyPdfDocumentParser


def create_pdf(*page_texts: str | None) -> bytes:
    writer = PdfWriter()

    for text in page_texts:
        page = writer.add_blank_page(width=300, height=300)
        if text is None:
            continue

        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        escaped_text = (
            text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({escaped_text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_parser_extracts_clean_text_and_preserves_page_numbers() -> None:
    pdf = create_pdf(None, "Warranty   coverage lasts two years.")

    pages = PyPdfDocumentParser().parse(pdf)

    assert len(pages) == 1
    assert pages[0].page_number == 2
    assert pages[0].content == "Warranty coverage lasts two years."


def test_parser_rejects_non_pdf_content() -> None:
    with pytest.raises(InvalidDocumentError, match="valid PDF"):
        PyPdfDocumentParser().parse(b"plain text")


def test_parser_reports_malformed_pdf() -> None:
    with pytest.raises(DocumentParsingError, match="Unable to parse"):
        PyPdfDocumentParser().parse(b"%PDF-not-a-real-document")


def test_parser_rejects_encrypted_pdf() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.encrypt("secret")
    output = BytesIO()
    writer.write(output)

    with pytest.raises(InvalidDocumentError, match="Encrypted"):
        PyPdfDocumentParser().parse(output.getvalue())
