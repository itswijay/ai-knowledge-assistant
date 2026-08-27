from app.domain.errors import DocumentProcessingError


class NoExtractableTextError(DocumentProcessingError):
    """A valid document contained no text that can be ingested."""
