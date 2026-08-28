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


class AccessTokenVerificationError(Exception):
    """Access-token verification could not run because its key source failed."""


class ResourceNotFoundError(Exception):
    """Base error for an application resource that cannot be exposed."""


class OrganizationNotFoundError(ResourceNotFoundError):
    """An organization does not exist or is not visible to the caller."""


class AssistantNotFoundError(ResourceNotFoundError):
    """An assistant does not exist or is not visible to the caller."""


class DocumentNotFoundError(ResourceNotFoundError):
    """A document does not exist or is not visible to the caller."""


class AuthorizationError(Exception):
    """Base error for an authenticated user who lacks required access."""


class MembershipRequiredError(AuthorizationError):
    """The authenticated user is not a member of the required organization."""


class InsufficientPermissionError(AuthorizationError):
    """The authenticated member's role cannot perform an operation."""


class ResourceConflictError(Exception):
    """A requested resource operation conflicts with persisted state."""


class TenantRepositoryError(Exception):
    """Tenant persistence failed without exposing database details."""


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
