from collections.abc import Sequence
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.domain.entities import DocumentPage
from app.domain.errors import DocumentParsingError, InvalidDocumentError
from app.infrastructure.documents.text_cleaner import clean_text
from app.infrastructure.documents.validation import has_pdf_signature


class PyPdfDocumentParser:
    def parse(self, content: bytes) -> Sequence[DocumentPage]:
        if not has_pdf_signature(content):
            raise InvalidDocumentError("File content is not a valid PDF")

        try:
            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise InvalidDocumentError("Encrypted PDFs are not supported")

            pages: list[DocumentPage] = []
            for page_number, page in enumerate(reader.pages, start=1):
                extracted_text = page.extract_text()
                if extracted_text is None:
                    continue
                cleaned_text = clean_text(extracted_text)
                if cleaned_text:
                    pages.append(
                        DocumentPage(
                            page_number=page_number,
                            content=cleaned_text,
                        )
                    )
            return tuple(pages)
        except PdfReadError as error:
            raise DocumentParsingError("Unable to parse PDF content") from error
