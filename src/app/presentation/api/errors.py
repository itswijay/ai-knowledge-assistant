from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.errors import (
    DocumentProcessingError,
    DocumentTooLargeError,
    EmbeddingGenerationError,
    VectorRepositoryError,
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


def register_exception_handlers(application: FastAPI) -> None:
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
