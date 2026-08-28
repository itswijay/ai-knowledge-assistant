from hashlib import sha256
from io import StringIO
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.core.config import get_database_settings

MIGRATION_0001_SHA256 = (
    "e1484ac87eef82f8aba0137e5b60250f94c2ef1a2da5f0a955bcbc0649e39feb"
)


def test_migration_history_has_one_expected_head() -> None:
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))

    assert scripts.get_heads() == ["0002"]
    assert scripts.get_base() == "0001"


def test_phase_one_migration_is_unchanged() -> None:
    migration_path = Path("alembic/versions/0001_create_document_vector_tables.py")

    assert sha256(migration_path.read_bytes()).hexdigest() == MIGRATION_0001_SHA256


def test_offline_migration_creates_pgvector_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@localhost/app",
    )
    get_database_settings.cache_clear()
    output = StringIO()
    config = Config("alembic.ini", output_buffer=output)

    try:
        command.upgrade(config, "head", sql=True)
    finally:
        get_database_settings.cache_clear()

    migration_sql = output.getvalue()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in migration_sql
    assert "CREATE TABLE documents" in migration_sql
    assert "CREATE TABLE document_chunks" in migration_sql
    assert migration_sql.count("original_filename VARCHAR(255) NOT NULL") == 1
    assert "ck_document_chunks_original_filename_not_blank" not in migration_sql
    assert "embedding VECTOR(768) NOT NULL" in migration_sql
    assert "ck_documents_original_filename_not_blank" in migration_sql
    assert "ck_documents_ck_documents" not in migration_sql


def test_offline_migration_adds_multi_tenant_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@localhost/app",
    )
    get_database_settings.cache_clear()
    output = StringIO()
    config = Config("alembic.ini", output_buffer=output)

    try:
        command.upgrade(config, "head", sql=True)
    finally:
        get_database_settings.cache_clear()

    migration_sql = output.getvalue()
    assert "CREATE TABLE organizations" in migration_sql
    assert "CREATE TABLE organization_members" in migration_sql
    assert "CREATE TABLE assistants" in migration_sql
    assert "PRIMARY KEY (organization_id, user_id)" in migration_sql
    assert "REFERENCES auth.users (id) ON DELETE CASCADE" in migration_sql
    assert "ck_organization_members_role_valid" in migration_sql
    assert "ix_organization_members_user_id" in migration_sql
    assert "ix_assistants_organization_id" in migration_sql
    assert "ix_documents_assistant_id" in migration_sql
    assert "ADD COLUMN assistant_id UUID NOT NULL" in migration_sql
    assert "fk_documents_assistant_id_assistants" in migration_sql
    assert migration_sql.index("DELETE FROM documents") < migration_sql.index(
        "ADD COLUMN assistant_id UUID NOT NULL"
    )
