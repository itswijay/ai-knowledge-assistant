from typing import Protocol
from uuid import UUID

from app.domain.entities.organization import OrganizationMember


class OrganizationMemberRepository(Protocol):
    async def get_membership(
        self,
        organization_id: UUID,
        user_id: UUID,
    ) -> OrganizationMember | None: ...
