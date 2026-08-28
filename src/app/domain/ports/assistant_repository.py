from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.domain.entities.assistant import Assistant


class AssistantRepository(Protocol):
    async def create(self, assistant: Assistant) -> None: ...

    async def list_by_organization(
        self,
        organization_id: UUID,
    ) -> Sequence[Assistant]: ...

    async def get_by_id(self, assistant_id: UUID) -> Assistant | None: ...

    async def update(self, assistant: Assistant) -> None: ...

    async def delete(self, assistant_id: UUID) -> bool: ...
