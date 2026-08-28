from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.domain.entities.document import Document


class DocumentRepository(Protocol):
    async def list_by_assistant(
        self,
        assistant_id: UUID,
    ) -> Sequence[Document]: ...

    async def get_by_id(self, document_id: UUID) -> Document | None: ...

    async def delete(self, document_id: UUID) -> bool: ...
