class AuthenticationError(Exception):
    """Base error for expected authentication failures."""


class MissingAccessTokenError(AuthenticationError):
    """An authenticated operation was requested without an access token."""


class InvalidAccessTokenError(AuthenticationError):
    """An access token failed validation or could not be trusted."""


class MalformedAccessTokenError(InvalidAccessTokenError):
    """An access token did not have a valid JWT structure."""


class ExpiredAccessTokenError(InvalidAccessTokenError):
    """An otherwise valid access token is no longer active."""


class DocumentProcessingError(Exception):
    """Base error for expected document-processing failures."""


class InvalidDocumentError(DocumentProcessingError):
    """The supplied document is unsupported or structurally invalid."""


class DocumentTooLargeError(InvalidDocumentError):
    """The supplied document exceeds the configured upload limit."""


class DocumentParsingError(DocumentProcessingError):
    """Text could not be extracted from an otherwise plausible document."""


class EmbeddingGenerationError(Exception):
    """An embedding provider could not produce valid vectors."""


class VectorRepositoryError(Exception):
    """Vector persistence or retrieval failed."""


class LLMGenerationError(Exception):
    """A language model could not produce a valid answer."""
