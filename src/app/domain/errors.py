class DocumentProcessingError(Exception):
    """Base error for expected document-processing failures."""


class InvalidDocumentError(DocumentProcessingError):
    """The supplied document is unsupported or structurally invalid."""


class DocumentTooLargeError(InvalidDocumentError):
    """The supplied document exceeds the configured upload limit."""


class DocumentParsingError(DocumentProcessingError):
    """Text could not be extracted from an otherwise plausible document."""
