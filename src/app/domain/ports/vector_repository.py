from collections.abc import Sequence
from typing import Protocol

from app.domain.entities import Document, DocumentChunk, RetrievedChunk
from app.domain.types import EmbeddingVector


class VectorRepository(Protocol):
    async def save_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None: ...

    async def search_similar(
        self,
        query_embedding: EmbeddingVector,
        *,
        limit: int,
        minimum_similarity: float,
    ) -> Sequence[RetrievedChunk]: ...
