from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncEngine

from app.application.services import WordChunker
from app.application.use_cases import IngestDocument
from app.core.config import Settings, get_settings
from app.infrastructure.ai import GeminiEmbeddingProvider
from app.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from app.infrastructure.database.vector_repository import PostgresVectorRepository
from app.infrastructure.documents import PdfUploadValidator, PyPdfDocumentParser


@dataclass(slots=True)
class ApplicationContainer:
    engine: AsyncEngine
    embedding_provider: GeminiEmbeddingProvider
    ingest_document: IngestDocument

    async def close(self) -> None:
        try:
            await self.embedding_provider.close()
        finally:
            await self.engine.dispose()


def build_application_container(settings: Settings) -> ApplicationContainer:
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)
    embedding_provider = GeminiEmbeddingProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_embedding_model,
        dimension=settings.embedding_dimension,
    )
    vector_repository = PostgresVectorRepository(
        session_factory,
        embedding_dimension=settings.embedding_dimension,
    )
    ingest_document = IngestDocument(
        validator=PdfUploadValidator(settings.max_upload_size_mb),
        parser=PyPdfDocumentParser(),
        chunker=WordChunker(),
        embedding_provider=embedding_provider,
        vector_repository=vector_repository,
    )
    return ApplicationContainer(
        engine=engine,
        embedding_provider=embedding_provider,
        ingest_document=ingest_document,
    )


def get_application_container(request: Request) -> ApplicationContainer:
    container: ApplicationContainer = request.app.state.container
    return container


def get_ingest_document(
    container: Annotated[ApplicationContainer, Depends(get_application_container)],
) -> IngestDocument:
    return container.ingest_document


def get_max_upload_size_bytes(
    settings: Annotated[Settings, Depends(get_settings)],
) -> int:
    return settings.max_upload_size_mb * 1024 * 1024
