from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.errors import (
    AccessTokenVerificationError,
    AuthenticationError,
    AuthorizationError,
    DocumentProcessingError,
    DocumentTooLargeError,
    EmbeddingGenerationError,
    ExpiredAccessTokenError,
    LLMGenerationError,
    MissingAccessTokenError,
    ResourceConflictError,
    ResourceNotFoundError,
    TenantRepositoryError,
    VectorRepositoryError,
)


async def authentication_handler(
    request: Request,
    error: AuthenticationError,
) -> JSONResponse:
    if isinstance(error, MissingAccessTokenError):
        detail = "Authentication credentials were not provided"
    elif isinstance(error, ExpiredAccessTokenError):
        detail = "Access token has expired"
    else:
        detail = "Access token is invalid"
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


async def access_token_verification_handler(
    request: Request,
    error: AccessTokenVerificationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Authentication service is temporarily unavailable"},
    )


async def resource_not_found_handler(
    request: Request,
    error: ResourceNotFoundError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(error) or "Resource not found"},
    )


async def authorization_handler(
    request: Request,
    error: AuthorizationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": str(error) or "Insufficient permission"},
    )


async def resource_conflict_handler(
    request: Request,
    error: ResourceConflictError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": "Resource conflicts with existing data"},
    )


async def tenant_repository_handler(
    request: Request,
    error: TenantRepositoryError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Data service is temporarily unavailable"},
    )


async def document_too_large_handler(
    request: Request,
    error: DocumentTooLargeError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        content={"detail": str(error)},
    )


async def document_processing_handler(
    request: Request,
    error: DocumentProcessingError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(error)},
    )


async def embedding_provider_handler(
    request: Request,
    error: EmbeddingGenerationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(error)},
    )


async def vector_repository_handler(
    request: Request,
    error: VectorRepositoryError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(error)},
    )


async def llm_provider_handler(
    request: Request,
    error: LLMGenerationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(error)},
    )


def register_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(
        AuthenticationError,
        authentication_handler,
    )
    application.add_exception_handler(
        AccessTokenVerificationError,
        access_token_verification_handler,
    )
    application.add_exception_handler(
        ResourceNotFoundError,
        resource_not_found_handler,
    )
    application.add_exception_handler(
        AuthorizationError,
        authorization_handler,
    )
    application.add_exception_handler(
        ResourceConflictError,
        resource_conflict_handler,
    )
    application.add_exception_handler(
        TenantRepositoryError,
        tenant_repository_handler,
    )
    application.add_exception_handler(
        DocumentTooLargeError,
        document_too_large_handler,
    )
    application.add_exception_handler(
        DocumentProcessingError,
        document_processing_handler,
    )
    application.add_exception_handler(
        EmbeddingGenerationError,
        embedding_provider_handler,
    )
    application.add_exception_handler(
        VectorRepositoryError,
        vector_repository_handler,
    )
    application.add_exception_handler(
        LLMGenerationError,
        llm_provider_handler,
    )
