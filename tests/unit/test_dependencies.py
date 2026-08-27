import pytest

from app.application.use_cases import IngestDocument
from app.core.config import Settings
from app.dependencies import build_application_container
from app.infrastructure.ai import GeminiEmbeddingProvider


@pytest.mark.asyncio
async def test_application_container_wires_and_closes_resources() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://postgres:password@localhost/app",
        gemini_api_key="test-api-key",
    )

    container = build_application_container(settings)

    assert isinstance(container.ingest_document, IngestDocument)
    assert isinstance(container.embedding_provider, GeminiEmbeddingProvider)
    assert container.engine.dialect.driver == "asyncpg"
    await container.close()
