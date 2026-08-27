from functools import lru_cache
from typing import Annotated

from pydantic import Field, PostgresDsn, Secret, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class Settings(BaseSettings):
    """Validated configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    database_url: Secret[PostgresDsn]
    gemini_api_key: SecretStr = Field(min_length=1)
    gemini_llm_model: NonEmptyString = "gemini-3.7-flash"
    gemini_embedding_model: NonEmptyString = "gemini-embedding-2"
    embedding_dimension: int = Field(default=768, gt=0)
    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_upload_size_mb: int = Field(default=10, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return one immutable settings instance per process."""

    return Settings()  # type: ignore[call-arg]
