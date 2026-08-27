from collections.abc import Sequence
from typing import Protocol

from app.domain.types import EmbeddingVector


class EmbeddingProvider(Protocol):
    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]: ...

    async def embed_query(self, text: str) -> EmbeddingVector: ...
