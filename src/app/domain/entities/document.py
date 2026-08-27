from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from uuid import UUID, uuid4

from app.domain.types import EmbeddingVector


@dataclass(frozen=True, slots=True)
class Document:
    original_filename: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.original_filename.strip():
            raise ValueError("original_filename must not be blank")


@dataclass(frozen=True, slots=True)
class DocumentPage:
    page_number: int
    content: str

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be at least 1")
        if not self.content.strip():
            raise ValueError("content must not be blank")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    document_id: UUID
    page_number: int
    chunk_index: int
    content: str
    embedding: EmbeddingVector
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be at least 1")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if not self.content.strip():
            raise ValueError("content must not be blank")
        if not self.embedding:
            raise ValueError("embedding must not be empty")
        if not all(isfinite(value) for value in self.embedding):
            raise ValueError("embedding values must be finite")
