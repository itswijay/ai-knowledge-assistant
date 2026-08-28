from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.application.services import OrganizationAccessChecker
from app.domain.entities import Organization, OrganizationMember, OrganizationRole
from app.domain.errors import OrganizationNotFoundError
from app.domain.ports import OrganizationRepository


@dataclass(frozen=True, slots=True)
class CreateOrganizationCommand:
    creator_user_id: UUID
    name: str


class CreateOrganization:
    def __init__(self, repository: OrganizationRepository) -> None:
        self._repository = repository

    async def execute(self, command: CreateOrganizationCommand) -> Organization:
        organization = Organization(name=command.name.strip())
        owner = OrganizationMember(
            organization_id=organization.id,
            user_id=command.creator_user_id,
            role=OrganizationRole.OWNER,
        )
        await self._repository.create_with_owner(organization, owner)
        return organization


class ListOrganizations:
    def __init__(self, repository: OrganizationRepository) -> None:
        self._repository = repository

    async def execute(self, user_id: UUID) -> Sequence[Organization]:
        return await self._repository.list_for_user(user_id)


class GetOrganization:
    def __init__(
        self,
        repository: OrganizationRepository,
        access_checker: OrganizationAccessChecker,
    ) -> None:
        self._repository = repository
        self._access_checker = access_checker

    async def execute(
        self,
        *,
        user_id: UUID,
        organization_id: UUID,
    ) -> Organization:
        await self._access_checker.require_member(
            user_id=user_id,
            organization_id=organization_id,
        )
        organization = await self._repository.get_by_id(organization_id)
        if organization is None:
            raise OrganizationNotFoundError("Organization not found")
        return organization
