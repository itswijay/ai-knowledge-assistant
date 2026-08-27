from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.entities import DocumentPage


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    page_number: int
    chunk_index: int
    content: str


@dataclass(frozen=True, slots=True)
class WordChunker:
    chunk_size_words: int = 500
    overlap_words: int = 100

    def __post_init__(self) -> None:
        if self.chunk_size_words < 1:
            raise ValueError("chunk_size_words must be at least 1")
        if self.overlap_words < 0:
            raise ValueError("overlap_words must not be negative")
        if self.overlap_words >= self.chunk_size_words:
            raise ValueError("overlap_words must be smaller than chunk_size_words")

    def chunk_pages(self, pages: Sequence[DocumentPage]) -> tuple[ChunkDraft, ...]:
        """Split each page independently while assigning document-wide indices."""

        chunks: list[ChunkDraft] = []
        step = self.chunk_size_words - self.overlap_words

        for page in pages:
            words = page.content.split()
            for start in range(0, len(words), step):
                end = min(start + self.chunk_size_words, len(words))
                chunks.append(
                    ChunkDraft(
                        page_number=page.page_number,
                        chunk_index=len(chunks),
                        content=" ".join(words[start:end]),
                    )
                )
                if end == len(words):
                    break

        return tuple(chunks)
