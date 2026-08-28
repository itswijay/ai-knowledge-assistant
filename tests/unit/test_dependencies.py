import pytest

from app.application.use_cases import AskQuestion, IngestDocument
from app.core.config import Settings
from app.dependencies import build_application_container
from app.infrastructure.ai import GeminiEmbeddingProvider, GeminiLLMProvider
from app.infrastructure.auth import SupabaseJWTVerifier


@pytest.mark.asyncio
async def test_application_container_wires_and_closes_resources() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://postgres:password@localhost/app",
        supabase_url="https://test-project.supabase.co",
        gemini_api_key="test-api-key",
    )

    container = build_application_container(settings)

    assert isinstance(container.ingest_document, IngestDocument)
    assert isinstance(container.ask_question, AskQuestion)
    assert isinstance(container.access_token_verifier, SupabaseJWTVerifier)
    assert isinstance(container.embedding_provider, GeminiEmbeddingProvider)
    assert isinstance(container.llm_provider, GeminiLLMProvider)
    assert container.engine.dialect.driver == "asyncpg"
    await container.close()
