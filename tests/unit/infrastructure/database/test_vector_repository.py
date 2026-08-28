from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import nan
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from app.domain.entities import Document, DocumentChunk
from app.domain.errors import VectorRepositoryError
from app.infrastructure.database.models import DocumentChunkModel, DocumentModel
from app.infrastructure.database.session import AsyncSessionFactory
from app.infrastructure.database.vector_repository import PostgresVectorRepository


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        return None


@dataclass
class FakeResult:
    rows: Sequence[tuple[DocumentChunkModel, str, float]]

    def all(self) -> Sequence[tuple[DocumentChunkModel, str, float]]:
        return self.rows


@dataclass
class FakeSession:
    rows: Sequence[tuple[DocumentChunkModel, str, float]] = ()
    execute_error: SQLAlchemyError | None = None
    added: list[object] = field(default_factory=list)
    executed_statement: object | None = None
    closed: bool = False

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        self.closed = True

    def begin(self) -> FakeTransaction:
        return FakeTransaction()

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def execute(self, statement: object) -> FakeResult:
        self.executed_statement = statement
        if self.execute_error is not None:
            raise self.execute_error
        return FakeResult(self.rows)


@dataclass
class FakeSessionFactory:
    session: FakeSession
    calls: int = 0

    def __call__(self) -> FakeSession:
        self.calls += 1
        return self.session


def build_repository(
    session: FakeSession,
    *,
    dimension: int = 3,
) -> PostgresVectorRepository:
    factory = FakeSessionFactory(session)
    return PostgresVectorRepository(
        cast(AsyncSessionFactory, factory),
        embedding_dimension=dimension,
    )


def build_chunk(document: Document) -> DocumentChunk:
    return DocumentChunk(
        document_id=document.id,
        page_number=2,
        chunk_index=0,
        content="Warranty coverage lasts two years.",
        embedding=(0.1, 0.2, 0.3),
    )


@pytest.mark.asyncio
async def test_save_document_maps_domain_entities_atomically() -> None:
    session = FakeSession()
    repository = build_repository(session)
    document = Document(assistant_id=uuid4(), original_filename="warranty.pdf")
    chunk = build_chunk(document)

    await repository.save_document(document, [chunk])

    assert session.closed is True
    assert len(session.added) == 1
    stored_document = session.added[0]
    assert isinstance(stored_document, DocumentModel)
    assert stored_document.id == document.id
    assert stored_document.assistant_id == document.assistant_id
    assert stored_document.original_filename == "warranty.pdf"
    assert len(stored_document.chunks) == 1
    assert stored_document.chunks[0].id == chunk.id
    assert not hasattr(stored_document.chunks[0], "original_filename")
    assert stored_document.chunks[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_save_document_rejects_inconsistent_chunk_metadata() -> None:
    session = FakeSession()
    repository = build_repository(session)
    document = Document(assistant_id=uuid4(), original_filename="warranty.pdf")
    wrong_document_chunk = DocumentChunk(
        document_id=uuid4(),
        page_number=1,
        chunk_index=0,
        content="Content",
        embedding=(0.1, 0.2, 0.3),
    )

    with pytest.raises(ValueError, match="belong"):
        await repository.save_document(document, [wrong_document_chunk])

    assert session.added == []


@pytest.mark.asyncio
async def test_search_filters_and_orders_inside_postgresql() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    stored_chunk = DocumentChunkModel(
        id=chunk_id,
        document_id=document_id,
        page_number=3,
        chunk_index=1,
        content="The warranty lasts two years.",
        embedding=[0.1, 0.2, 0.3],
    )
    session = FakeSession(rows=[(stored_chunk, "warranty.pdf", 0.91)])
    repository = build_repository(session)

    retrieved = await repository.search_similar(
        (0.1, 0.2, 0.3),
        limit=5,
        minimum_similarity=0.75,
    )

    assert len(retrieved) == 1
    assert retrieved[0].chunk_id == chunk_id
    assert retrieved[0].document_id == document_id
    assert retrieved[0].original_filename == "warranty.pdf"
    assert retrieved[0].page_number == 3
    assert retrieved[0].similarity_score == 0.91
    statement = session.executed_statement
    assert statement is not None
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "JOIN documents" in compiled_sql
    assert "documents.original_filename" in compiled_sql
    assert "document_chunks.embedding <=>" in compiled_sql
    assert "WHERE" in compiled_sql
    assert ">=" in compiled_sql
    assert "ORDER BY" in compiled_sql
    assert "LIMIT" in compiled_sql


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding", "limit", "threshold"),
    [
        ((0.1, 0.2), 5, 0.7),
        ((0.0, 0.0, 0.0), 5, 0.7),
        ((0.1, 0.2, nan), 5, 0.7),
        ((0.1, 0.2, 0.3), 0, 0.7),
        ((0.1, 0.2, 0.3), 51, 0.7),
        ((0.1, 0.2, 0.3), 5, -0.1),
        ((0.1, 0.2, 0.3), 5, 1.1),
    ],
)
async def test_search_rejects_invalid_parameters(
    embedding: tuple[float, ...],
    limit: int,
    threshold: float,
) -> None:
    session = FakeSession()
    repository = build_repository(session)

    with pytest.raises(ValueError):
        await repository.search_similar(
            embedding,
            limit=limit,
            minimum_similarity=threshold,
        )

    assert session.executed_statement is None


@pytest.mark.asyncio
async def test_database_errors_are_wrapped_without_query_details() -> None:
    session = FakeSession(execute_error=SQLAlchemyError("database unavailable"))
    repository = build_repository(session)

    with pytest.raises(VectorRepositoryError, match="Unable to retrieve"):
        await repository.search_similar(
            (0.1, 0.2, 0.3),
            limit=5,
            minimum_similarity=0.7,
        )
