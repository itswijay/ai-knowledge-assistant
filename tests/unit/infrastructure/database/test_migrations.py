from io import StringIO

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.core.config import get_database_settings


def test_migration_history_has_one_expected_head() -> None:
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))

    assert scripts.get_heads() == ["0001"]
    assert scripts.get_base() == "0001"


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
    assert "embedding VECTOR NOT NULL" in migration_sql
    assert "ck_documents_original_filename_not_blank" in migration_sql
    assert "ck_documents_ck_documents" not in migration_sql
