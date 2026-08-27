from typing import Any

import pytest
from pydantic import ValidationError

from app.core.config import DatabaseSettings, Settings
from app.core.constants import EMBEDDING_DIMENSION

REQUIRED_SETTINGS: dict[str, Any] = {
    "database_url": "postgresql+asyncpg://postgres:password@localhost:5432/app",
    "gemini_api_key": "test-api-key",
}


def build_settings(**overrides: Any) -> Settings:
    values = REQUIRED_SETTINGS | overrides
    return Settings(_env_file=None, **values)


def test_settings_use_safe_phase_one_defaults() -> None:
    settings = build_settings()

    assert settings.gemini_llm_model == "gemini-3.7-flash"
    assert settings.gemini_max_output_tokens == 512
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.embedding_dimension == EMBEDDING_DIMENSION == 768
    assert settings.rag_top_k == 5
    assert settings.rag_similarity_threshold == 0.7
    assert settings.max_upload_size_mb == 10


def test_credentials_are_masked_in_settings_representation() -> None:
    settings = build_settings(
        database_url="postgresql+asyncpg://user:database-secret@localhost/app",
        gemini_api_key="api-secret",
    )

    settings_representation = repr(settings)
    assert "database-secret" not in settings_representation
    assert "api-secret" not in settings_representation
    assert settings.gemini_api_key.get_secret_value() == "api-secret"


def test_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "DATABASE_URL": "postgresql+asyncpg://postgres:password@localhost/app",
        "GEMINI_API_KEY": "environment-api-key",
        "GEMINI_LLM_MODEL": "test-llm-model",
        "GEMINI_MAX_OUTPUT_TOKENS": "700",
        "GEMINI_EMBEDDING_MODEL": "test-embedding-model",
        "EMBEDDING_DIMENSION": "768",
        "RAG_TOP_K": "8",
        "RAG_SIMILARITY_THRESHOLD": "0.82",
        "MAX_UPLOAD_SIZE_MB": "15",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.gemini_llm_model == "test-llm-model"
    assert settings.gemini_max_output_tokens == 700
    assert settings.gemini_embedding_model == "test-embedding-model"
    assert settings.embedding_dimension == EMBEDDING_DIMENSION
    assert settings.rag_top_k == 8
    assert settings.rag_similarity_threshold == 0.82
    assert settings.max_upload_size_mb == 15


def test_database_settings_do_not_require_provider_credentials() -> None:
    settings = DatabaseSettings(
        _env_file=None,
        database_url=REQUIRED_SETTINGS["database_url"],
    )

    assert settings.database_url.get_secret_value().scheme == "postgresql+asyncpg"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("database_url", "sqlite:///local.db"),
        ("database_url", "postgresql://postgres:password@localhost/app"),
        ("gemini_llm_model", "   "),
        ("gemini_max_output_tokens", 0),
        ("gemini_max_output_tokens", 8193),
        ("embedding_dimension", 767),
        ("embedding_dimension", 769),
        ("rag_top_k", 0),
        ("rag_top_k", 51),
        ("rag_similarity_threshold", -0.01),
        ("rag_similarity_threshold", 1.01),
        ("max_upload_size_mb", 0),
    ],
)
def test_settings_reject_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        build_settings(**{field: value})


def test_required_settings_must_be_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)  # type: ignore[call-arg]

    missing_fields = {item["loc"][0] for item in error.value.errors()}
    assert missing_fields == {"database_url", "gemini_api_key"}


def test_environment_cannot_override_schema_embedding_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", str(REQUIRED_SETTINGS["database_url"]))
    monkeypatch.setenv("GEMINI_API_KEY", str(REQUIRED_SETTINGS["gemini_api_key"]))
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)  # type: ignore[call-arg]

    assert error.value.errors()[0]["loc"] == ("embedding_dimension",)
