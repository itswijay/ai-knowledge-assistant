from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.domain.entities.organization import Organization, OrganizationMember


class OrganizationRepository(Protocol):
    async def create_with_owner(
        self,
        organization: Organization,
        owner: OrganizationMember,
    ) -> None:
        """Persist an organization and its owner membership atomically."""

        ...

    async def list_for_user(self, user_id: UUID) -> Sequence[Organization]: ...

    async def get_by_id(self, organization_id: UUID) -> Organization | None: ...
