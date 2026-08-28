import os
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.constants import EMBEDDING_DIMENSION
from app.domain.entities import Document, DocumentChunk
from app.infrastructure.database.models import AssistantModel, OrganizationModel
from app.infrastructure.database.vector_repository import PostgresVectorRepository


def _integration_database_url() -> str:
    if os.getenv("RUN_POSTGRES_INTEGRATION_TESTS") != "1":
        pytest.skip("set RUN_POSTGRES_INTEGRATION_TESTS=1 to run PostgreSQL tests")

    test_database_url = os.getenv("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL tests")
    if test_database_url == os.getenv("DATABASE_URL"):
        pytest.fail("TEST_DATABASE_URL must differ from DATABASE_URL")
    if not test_database_url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEST_DATABASE_URL must use postgresql+asyncpg")
    return test_database_url


def _embedding(*, first: float, second: float) -> tuple[float, ...]:
    return (first, second, *(0.0 for _ in range(EMBEDDING_DIMENSION - 2)))


@pytest.mark.asyncio
async def test_postgresql_vector_search_never_crosses_assistant_boundary() -> None:
    engine = create_async_engine(_integration_database_url(), pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    organization_a_id = uuid4()
    organization_b_id = uuid4()
    assistant_a_id = uuid4()
    assistant_b_id = uuid4()
    setup_committed = False

    try:
        async with session_factory() as session:
            async with session.begin():
                session.add_all(
                    [
                        OrganizationModel(
                            id=organization_a_id,
                            name="Tenant isolation test A",
                            assistants=[
                                AssistantModel(
                                    id=assistant_a_id,
                                    name="Assistant A",
                                )
                            ],
                        ),
                        OrganizationModel(
                            id=organization_b_id,
                            name="Tenant isolation test B",
                            assistants=[
                                AssistantModel(
                                    id=assistant_b_id,
                                    name="Assistant B",
                                )
                            ],
                        ),
                    ]
                )
        setup_committed = True

        repository = PostgresVectorRepository(
            session_factory,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        document_a = Document(
            assistant_id=assistant_a_id,
            original_filename="assistant-a.pdf",
        )
        document_b = Document(
            assistant_id=assistant_b_id,
            original_filename="assistant-b-private.pdf",
        )
        await repository.save_document(
            document_a,
            [
                DocumentChunk(
                    document_id=document_a.id,
                    page_number=1,
                    chunk_index=0,
                    content="Assistant A private content",
                    embedding=_embedding(first=0.8, second=0.6),
                )
            ],
        )
        await repository.save_document(
            document_b,
            [
                DocumentChunk(
                    document_id=document_b.id,
                    page_number=1,
                    chunk_index=0,
                    content="Assistant B higher-scoring private content",
                    embedding=_embedding(first=1.0, second=0.0),
                )
            ],
        )

        query_embedding = _embedding(first=1.0, second=0.0)
        assistant_a_results = await repository.search_similar(
            assistant_a_id,
            query_embedding,
            limit=5,
            minimum_similarity=0.0,
        )
        assistant_b_results = await repository.search_similar(
            assistant_b_id,
            query_embedding,
            limit=5,
            minimum_similarity=0.0,
        )

        assert [chunk.document_id for chunk in assistant_a_results] == [document_a.id]
        assert [chunk.document_id for chunk in assistant_b_results] == [document_b.id]
        assert all(
            chunk.original_filename != "assistant-b-private.pdf"
            for chunk in assistant_a_results
        )
        assert all(
            chunk.original_filename != "assistant-a.pdf"
            for chunk in assistant_b_results
        )
    finally:
        if setup_committed:
            async with session_factory() as session:
                async with session.begin():
                    await session.execute(
                        delete(OrganizationModel).where(
                            OrganizationModel.id.in_(
                                [organization_a_id, organization_b_id]
                            )
                        )
                    )
        await engine.dispose()
