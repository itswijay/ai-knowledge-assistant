from functools import lru_cache
from typing import Annotated

from pydantic import (
    Field,
    PostgresDsn,
    Secret,
    SecretStr,
    StringConstraints,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import EMBEDDING_DIMENSION

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class DatabaseSettings(BaseSettings):
    """Database-only settings used by runtime code and migrations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    database_url: Secret[PostgresDsn]

    @field_validator("database_url")
    @classmethod
    def require_asyncpg_driver(
        cls,
        value: Secret[PostgresDsn],
    ) -> Secret[PostgresDsn]:
        if value.get_secret_value().scheme != "postgresql+asyncpg":
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg scheme")
        return value


class Settings(DatabaseSettings):
    """Validated configuration loaded from environment variables."""

    gemini_api_key: SecretStr = Field(min_length=1)
    gemini_llm_model: NonEmptyString = "gemini-3.7-flash"
    gemini_max_output_tokens: int = Field(default=512, ge=1, le=8192)
    gemini_embedding_model: NonEmptyString = "gemini-embedding-2"
    embedding_dimension: int = Field(
        default=EMBEDDING_DIMENSION,
        ge=EMBEDDING_DIMENSION,
        le=EMBEDDING_DIMENSION,
    )
    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_upload_size_mb: int = Field(default=10, gt=0)


@lru_cache
def get_database_settings() -> DatabaseSettings:
    """Return database settings without requiring unrelated provider secrets."""

    return DatabaseSettings()  # type: ignore[call-arg]


@lru_cache
def get_settings() -> Settings:
    """Return one immutable settings instance per process."""

    return Settings()  # type: ignore[call-arg]
