from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncEngine

from app.application.services import GroundedPromptBuilder, WordChunker
from app.application.use_cases import AskQuestion, IngestDocument
from app.core.config import Settings, get_settings
from app.infrastructure.ai import GeminiEmbeddingProvider, GeminiLLMProvider
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
    llm_provider: GeminiLLMProvider
    ingest_document: IngestDocument
    ask_question: AskQuestion

    async def close(self) -> None:
        try:
            await self.embedding_provider.close()
        finally:
            try:
                await self.llm_provider.close()
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
    llm_provider = GeminiLLMProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_llm_model,
        max_output_tokens=settings.gemini_max_output_tokens,
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
    ask_question = AskQuestion(
        embedding_provider=embedding_provider,
        vector_repository=vector_repository,
        llm_provider=llm_provider,
        prompt_builder=GroundedPromptBuilder(),
        top_k=settings.rag_top_k,
        similarity_threshold=settings.rag_similarity_threshold,
    )
    return ApplicationContainer(
        engine=engine,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        ingest_document=ingest_document,
        ask_question=ask_question,
    )


def get_application_container(request: Request) -> ApplicationContainer:
    container: ApplicationContainer = request.app.state.container
    return container


def get_ingest_document(
    container: Annotated[ApplicationContainer, Depends(get_application_container)],
) -> IngestDocument:
    return container.ingest_document


def get_ask_question(
    container: Annotated[ApplicationContainer, Depends(get_application_container)],
) -> AskQuestion:
    return container.ask_question


def get_max_upload_size_bytes(
    settings: Annotated[Settings, Depends(get_settings)],
) -> int:
    return settings.max_upload_size_mb * 1024 * 1024
