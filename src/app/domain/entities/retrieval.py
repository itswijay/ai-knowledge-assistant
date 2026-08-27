from dataclasses import dataclass
from math import isfinite
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    original_filename: str
    page_number: int
    chunk_index: int
    content: str
    similarity_score: float

    def __post_init__(self) -> None:
        if not self.original_filename.strip():
            raise ValueError("original_filename must not be blank")
        if self.page_number < 1:
            raise ValueError("page_number must be at least 1")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if not self.content.strip():
            raise ValueError("content must not be blank")
        if not isfinite(self.similarity_score):
            raise ValueError("similarity_score must be finite")
        if not -1.0 <= self.similarity_score <= 1.0:
            raise ValueError("similarity_score must be between -1 and 1")
