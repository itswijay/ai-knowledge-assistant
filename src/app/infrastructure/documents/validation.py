import re
import unicodedata
from dataclasses import dataclass

from app.domain.errors import DocumentTooLargeError, InvalidDocumentError

PDF_SIGNATURE = b"%PDF-"
UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^\w .-]", flags=re.UNICODE)
REPEATED_UNDERSCORES = re.compile(r"_+")
MAX_FILENAME_LENGTH = 255


def has_pdf_signature(content: bytes) -> bool:
    return content.startswith(PDF_SIGNATURE)


def sanitize_pdf_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKC", filename).replace("\\", "/")
    basename = normalized.rsplit("/", maxsplit=1)[-1].strip()
    if not basename or not basename.casefold().endswith(".pdf"):
        raise InvalidDocumentError("A filename with a .pdf extension is required")

    stem = basename[:-4]
    safe_stem = UNSAFE_FILENAME_CHARACTERS.sub("_", stem)
    safe_stem = REPEATED_UNDERSCORES.sub("_", safe_stem).strip(" ._-")
    if not safe_stem:
        safe_stem = "document"

    maximum_stem_length = MAX_FILENAME_LENGTH - len(".pdf")
    return f"{safe_stem[:maximum_stem_length]}.pdf"


@dataclass(frozen=True, slots=True)
class PdfUploadValidator:
    max_upload_size_mb: int

    def __post_init__(self) -> None:
        if self.max_upload_size_mb < 1:
            raise ValueError("max_upload_size_mb must be at least 1")

    def validate(self, filename: str, content: bytes) -> str:
        """Validate PDF identity and size, returning a safe filename."""

        safe_filename = sanitize_pdf_filename(filename)
        maximum_size_bytes = self.max_upload_size_mb * 1024 * 1024
        if len(content) > maximum_size_bytes:
            raise DocumentTooLargeError(
                f"PDF exceeds the {self.max_upload_size_mb} MB upload limit"
            )
        if not has_pdf_signature(content):
            raise InvalidDocumentError("File content is not a valid PDF")
        return safe_filename
