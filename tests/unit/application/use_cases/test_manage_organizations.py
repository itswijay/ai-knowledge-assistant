from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from app.application.services import OrganizationAccessChecker
from app.application.use_cases import (
    CreateOrganization,
    CreateOrganizationCommand,
    GetOrganization,
    ListOrganizations,
)
from app.domain.entities import Organization, OrganizationMember, OrganizationRole
from app.domain.errors import OrganizationNotFoundError


@dataclass
class FakeOrganizationRepository:
    organizations: dict[UUID, Organization] = field(default_factory=dict)
    listed: Sequence[Organization] = ()
    created: list[tuple[Organization, OrganizationMember]] = field(default_factory=list)
    list_calls: list[UUID] = field(default_factory=list)
    get_calls: list[UUID] = field(default_factory=list)

    async def create_with_owner(
        self,
        organization: Organization,
        owner: OrganizationMember,
    ) -> None:
        self.created.append((organization, owner))

    async def list_for_user(self, user_id: UUID) -> Sequence[Organization]:
        self.list_calls.append(user_id)
        return self.listed

    async def get_by_id(self, organization_id: UUID) -> Organization | None:
        self.get_calls.append(organization_id)
        return self.organizations.get(organization_id)


@dataclass
class FakeMembershipRepository:
    memberships: dict[tuple[UUID, UUID], OrganizationMember]
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def get_membership(
        self,
        organization_id: UUID,
        user_id: UUID,
    ) -> OrganizationMember | None:
        self.calls.append((organization_id, user_id))
        return self.memberships.get((organization_id, user_id))


def make_membership(
    *,
    organization_id: UUID,
    user_id: UUID,
    role: OrganizationRole = OrganizationRole.MEMBER,
) -> OrganizationMember:
    return OrganizationMember(
        organization_id=organization_id,
        user_id=user_id,
        role=role,
    )


@pytest.mark.asyncio
async def test_create_organization_persists_creator_as_owner_atomically() -> None:
    repository = FakeOrganizationRepository()
    creator_user_id = uuid4()

    organization = await CreateOrganization(repository).execute(
        CreateOrganizationCommand(
            creator_user_id=creator_user_id,
            name="  Acme Support  ",
        )
    )

    assert organization.name == "Acme Support"
    assert len(repository.created) == 1
    persisted_organization, persisted_owner = repository.created[0]
    assert persisted_organization == organization
    assert persisted_owner.organization_id == organization.id
    assert persisted_owner.user_id == creator_user_id
    assert persisted_owner.role is OrganizationRole.OWNER


@pytest.mark.asyncio
async def test_create_organization_rejects_blank_name_before_persistence() -> None:
    repository = FakeOrganizationRepository()

    with pytest.raises(ValueError, match="name must not be blank"):
        await CreateOrganization(repository).execute(
            CreateOrganizationCommand(creator_user_id=uuid4(), name="   ")
        )

    assert repository.created == []


@pytest.mark.asyncio
async def test_list_organizations_uses_authenticated_user_scope() -> None:
    user_id = uuid4()
    expected = (
        Organization(name="Acme"),
        Organization(name="Example"),
    )
    repository = FakeOrganizationRepository(listed=expected)

    organizations = await ListOrganizations(repository).execute(user_id)

    assert organizations == expected
    assert repository.list_calls == [user_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("role", list(OrganizationRole))
async def test_get_organization_allows_every_member_role(
    role: OrganizationRole,
) -> None:
    user_id = uuid4()
    organization = Organization(name="Acme")
    repository = FakeOrganizationRepository(
        organizations={organization.id: organization}
    )
    membership_repository = FakeMembershipRepository(
        {
            (organization.id, user_id): make_membership(
                organization_id=organization.id,
                user_id=user_id,
                role=role,
            )
        }
    )
    use_case = GetOrganization(
        repository,
        OrganizationAccessChecker(membership_repository),
    )

    actual = await use_case.execute(
        user_id=user_id,
        organization_id=organization.id,
    )

    assert actual == organization
    assert membership_repository.calls == [(organization.id, user_id)]
    assert repository.get_calls == [organization.id]


@pytest.mark.asyncio
async def test_get_organization_conceals_cross_tenant_access() -> None:
    user_id = uuid4()
    organization = Organization(name="Private")
    repository = FakeOrganizationRepository(
        organizations={organization.id: organization}
    )
    use_case = GetOrganization(
        repository,
        OrganizationAccessChecker(FakeMembershipRepository({})),
    )

    with pytest.raises(OrganizationNotFoundError, match="^Organization not found$"):
        await use_case.execute(
            user_id=user_id,
            organization_id=organization.id,
        )

    assert repository.get_calls == []


@pytest.mark.asyncio
async def test_get_organization_handles_missing_record_after_access_check() -> None:
    user_id = uuid4()
    organization_id = uuid4()
    membership_repository = FakeMembershipRepository(
        {
            (organization_id, user_id): make_membership(
                organization_id=organization_id,
                user_id=user_id,
            )
        }
    )
    repository = FakeOrganizationRepository()
    use_case = GetOrganization(
        repository,
        OrganizationAccessChecker(membership_repository),
    )

    with pytest.raises(OrganizationNotFoundError, match="^Organization not found$"):
        await use_case.execute(
            user_id=user_id,
            organization_id=organization_id,
        )

    assert repository.get_calls == [organization_id]
