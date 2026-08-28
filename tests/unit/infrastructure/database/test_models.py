from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.core.constants import EMBEDDING_DIMENSION
from app.infrastructure.database.base import Base
from app.infrastructure.database.models import (
    AssistantModel,
    DocumentChunkModel,
    DocumentModel,
    OrganizationMemberModel,
    OrganizationModel,
)


def test_expected_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {
        "organizations",
        "organization_members",
        "assistants",
        "documents",
        "document_chunks",
    }


def test_organization_membership_uses_composite_identity_and_role_check() -> None:
    table = OrganizationMemberModel.__table__
    primary_key = next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, PrimaryKeyConstraint)
    )
    check_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert [column.name for column in primary_key.columns] == [
        "organization_id",
        "user_id",
    ]
    assert check_names == {"ck_organization_members_role_valid"}
    assert {index.name for index in table.indexes} == {
        "ix_organization_members_organization_id",
        "ix_organization_members_user_id",
    }


def test_tenant_ownership_foreign_keys_cascade() -> None:
    membership_foreign_key = next(
        iter(OrganizationMemberModel.__table__.c.organization_id.foreign_keys)
    )
    assistant_foreign_key = next(
        iter(AssistantModel.__table__.c.organization_id.foreign_keys)
    )
    document_foreign_key = next(
        iter(DocumentModel.__table__.c.assistant_id.foreign_keys)
    )

    assert membership_foreign_key.target_fullname == "organizations.id"
    assert membership_foreign_key.ondelete == "CASCADE"
    assert assistant_foreign_key.target_fullname == "organizations.id"
    assert assistant_foreign_key.ondelete == "CASCADE"
    assert document_foreign_key.target_fullname == "assistants.id"
    assert document_foreign_key.ondelete == "CASCADE"


def test_document_model_has_required_assistant_scope() -> None:
    assistant_id = DocumentModel.__table__.c.assistant_id

    assert assistant_id.nullable is False
    assert {index.name for index in DocumentModel.__table__.indexes} == {
        "ix_documents_assistant_id"
    }


def test_assistant_model_matches_bounded_customization_schema() -> None:
    columns = AssistantModel.__table__.columns

    assert columns.name.type.length == 100
    assert columns.description.type.length == 1000
    assert columns.welcome_message.type.length == 500
    assert columns.assistant_instructions.type.length == 4000
    assert columns.logo_url.type.length == 2048
    assert columns.primary_color.type.length == 7
    assert {index.name for index in AssistantModel.__table__.indexes} == {
        "ix_assistants_organization_id"
    }


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
    organization_ddl = str(
        CreateTable(OrganizationModel.__table__).compile(dialect=postgresql.dialect())
    )
    membership_ddl = str(
        CreateTable(OrganizationMemberModel.__table__).compile(
            dialect=postgresql.dialect()
        )
    )
    assistant_ddl = str(
        CreateTable(AssistantModel.__table__).compile(dialect=postgresql.dialect())
    )
    document_ddl = str(
        CreateTable(DocumentModel.__table__).compile(dialect=postgresql.dialect())
    )
    chunk_ddl = str(
        CreateTable(DocumentChunkModel.__table__).compile(dialect=postgresql.dialect())
    )

    assert "CREATE TABLE documents" in document_ddl
    assert "CREATE TABLE document_chunks" in chunk_ddl
    assert "original_filename VARCHAR(255) NOT NULL" in document_ddl
    assert "assistant_id UUID NOT NULL" in document_ddl
    assert "original_filename" not in chunk_ddl
    assert "embedding VECTOR(768) NOT NULL" in chunk_ddl
    assert "CREATE TABLE organizations" in organization_ddl
    assert "CREATE TABLE organization_members" in membership_ddl
    assert "PRIMARY KEY (organization_id, user_id)" in membership_ddl
    assert "CREATE TABLE assistants" in assistant_ddl
