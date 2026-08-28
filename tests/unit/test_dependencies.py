import pytest

from app.application.use_cases import (
    AskQuestion,
    CreateAssistant,
    CreateOrganization,
    DeleteAssistant,
    DeleteDocument,
    GetAssistant,
    GetOrganization,
    IngestDocument,
    ListAssistants,
    ListDocuments,
    ListOrganizations,
    UpdateAssistant,
)
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
    assert isinstance(container.create_organization, CreateOrganization)
    assert isinstance(container.list_organizations, ListOrganizations)
    assert isinstance(container.get_organization, GetOrganization)
    assert isinstance(container.create_assistant, CreateAssistant)
    assert isinstance(container.list_assistants, ListAssistants)
    assert isinstance(container.get_assistant, GetAssistant)
    assert isinstance(container.update_assistant, UpdateAssistant)
    assert isinstance(container.delete_assistant, DeleteAssistant)
    assert isinstance(container.list_documents, ListDocuments)
    assert isinstance(container.delete_document, DeleteDocument)
    assert isinstance(container.access_token_verifier, SupabaseJWTVerifier)
    assert isinstance(container.embedding_provider, GeminiEmbeddingProvider)
    assert isinstance(container.llm_provider, GeminiLLMProvider)
    assert container.engine.dialect.driver == "asyncpg"
    await container.close()
