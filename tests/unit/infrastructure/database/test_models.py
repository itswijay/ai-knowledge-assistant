from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.core.constants import EMBEDDING_DIMENSION
from app.infrastructure.database.base import Base
from app.infrastructure.database.models import DocumentChunkModel, DocumentModel


def test_expected_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {"documents", "document_chunks"}


def test_document_chunk_preserves_required_metadata() -> None:
    columns = DocumentChunkModel.__table__.columns

    assert {
        "id",
        "document_id",
        "page_number",
        "chunk_index",
        "content",
        "embedding",
        "created_at",
    } == set(columns.keys())
    assert all(not column.nullable for column in columns)


def test_embedding_uses_schema_fixed_pgvector_dimension() -> None:
    embedding_type = DocumentChunkModel.__table__.c.embedding.type

    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == EMBEDDING_DIMENSION == 768


def test_chunk_ownership_cascades_on_document_deletion() -> None:
    foreign_key = next(iter(DocumentChunkModel.__table__.c.document_id.foreign_keys))

    assert foreign_key.target_fullname == "documents.id"
    assert foreign_key.ondelete == "CASCADE"


def test_schema_enforces_chunk_metadata_invariants() -> None:
    constraints = DocumentChunkModel.__table__.constraints
    check_names = {
        constraint.name
        for constraint in constraints
        if isinstance(constraint, CheckConstraint)
    }
    unique_constraints = [
        constraint
        for constraint in constraints
        if isinstance(constraint, UniqueConstraint)
    ]

    assert check_names == {
        "ck_document_chunks_page_number_positive",
        "ck_document_chunks_chunk_index_not_negative",
        "ck_document_chunks_content_not_blank",
    }
    assert len(unique_constraints) == 1
    assert [column.name for column in unique_constraints[0].columns] == [
        "document_id",
        "chunk_index",
    ]


def test_models_compile_to_postgresql_vector_schema() -> None:
    document_ddl = str(
        CreateTable(DocumentModel.__table__).compile(dialect=postgresql.dialect())
    )
    chunk_ddl = str(
        CreateTable(DocumentChunkModel.__table__).compile(dialect=postgresql.dialect())
    )

    assert "CREATE TABLE documents" in document_ddl
    assert "CREATE TABLE document_chunks" in chunk_ddl
    assert "original_filename VARCHAR(255) NOT NULL" in document_ddl
    assert "original_filename" not in chunk_ddl
    assert "embedding VECTOR(768) NOT NULL" in chunk_ddl
