from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from app.application.services import (
    AssistantAccessChecker,
    OrganizationAccessChecker,
)
from app.domain.entities import Assistant, OrganizationMember, OrganizationRole
from app.domain.errors import (
    AssistantNotFoundError,
    InsufficientPermissionError,
    OrganizationNotFoundError,
)


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


@dataclass
class FakeAssistantRepository:
    assistants: dict[UUID, Assistant]
    calls: list[UUID] = field(default_factory=list)

    async def get_by_id(self, assistant_id: UUID) -> Assistant | None:
        self.calls.append(assistant_id)
        return self.assistants.get(assistant_id)


def membership(
    *,
    organization_id: UUID,
    user_id: UUID,
    role: OrganizationRole,
) -> OrganizationMember:
    return OrganizationMember(
        organization_id=organization_id,
        user_id=user_id,
        role=role,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", list(OrganizationRole))
async def test_organization_member_access_accepts_every_role(
    role: OrganizationRole,
) -> None:
    organization_id = uuid4()
    user_id = uuid4()
    expected = membership(
        organization_id=organization_id,
        user_id=user_id,
        role=role,
    )
    repository = FakeMembershipRepository({(organization_id, user_id): expected})

    actual = await OrganizationAccessChecker(repository).require_member(
        user_id=user_id,
        organization_id=organization_id,
    )

    assert actual == expected
    assert repository.calls == [(organization_id, user_id)]


@pytest.mark.asyncio
async def test_organization_access_conceals_non_membership_as_not_found() -> None:
    organization_id = uuid4()
    user_id = uuid4()
    repository = FakeMembershipRepository({})

    with pytest.raises(OrganizationNotFoundError, match="^Organization not found$"):
        await OrganizationAccessChecker(repository).require_member(
            user_id=user_id,
            organization_id=organization_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [OrganizationRole.OWNER, OrganizationRole.ADMIN],
)
async def test_organization_manager_access_accepts_privileged_roles(
    role: OrganizationRole,
) -> None:
    organization_id = uuid4()
    user_id = uuid4()
    expected = membership(
        organization_id=organization_id,
        user_id=user_id,
        role=role,
    )
    repository = FakeMembershipRepository({(organization_id, user_id): expected})

    actual = await OrganizationAccessChecker(repository).require_manager(
        user_id=user_id,
        organization_id=organization_id,
    )

    assert actual == expected


@pytest.mark.asyncio
async def test_organization_manager_access_rejects_member_role() -> None:
    organization_id = uuid4()
    user_id = uuid4()
    repository = FakeMembershipRepository(
        {
            (organization_id, user_id): membership(
                organization_id=organization_id,
                user_id=user_id,
                role=OrganizationRole.MEMBER,
            )
        }
    )

    with pytest.raises(
        InsufficientPermissionError,
        match="^Owner or admin role required$",
    ):
        await OrganizationAccessChecker(repository).require_manager(
            user_id=user_id,
            organization_id=organization_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", list(OrganizationRole))
async def test_assistant_member_access_accepts_every_role(
    role: OrganizationRole,
) -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    assistant_repository = FakeAssistantRepository({assistant.id: assistant})
    membership_repository = FakeMembershipRepository(
        {
            (assistant.organization_id, user_id): membership(
                organization_id=assistant.organization_id,
                user_id=user_id,
                role=role,
            )
        }
    )

    actual = await AssistantAccessChecker(
        assistant_repository,
        membership_repository,
    ).require_member(user_id=user_id, assistant_id=assistant.id)

    assert actual == assistant
    assert assistant_repository.calls == [assistant.id]
    assert membership_repository.calls == [(assistant.organization_id, user_id)]


@pytest.mark.asyncio
async def test_assistant_access_conceals_missing_assistant() -> None:
    assistant_id = uuid4()
    membership_repository = FakeMembershipRepository({})

    with pytest.raises(AssistantNotFoundError, match="^Assistant not found$"):
        await AssistantAccessChecker(
            FakeAssistantRepository({}),
            membership_repository,
        ).require_member(user_id=uuid4(), assistant_id=assistant_id)

    assert membership_repository.calls == []


@pytest.mark.asyncio
async def test_assistant_access_conceals_cross_tenant_request() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    membership_repository = FakeMembershipRepository({})

    with pytest.raises(AssistantNotFoundError, match="^Assistant not found$"):
        await AssistantAccessChecker(
            FakeAssistantRepository({assistant.id: assistant}),
            membership_repository,
        ).require_member(user_id=user_id, assistant_id=assistant.id)

    assert membership_repository.calls == [(assistant.organization_id, user_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [OrganizationRole.OWNER, OrganizationRole.ADMIN],
)
async def test_assistant_manager_access_accepts_privileged_roles(
    role: OrganizationRole,
) -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    membership_repository = FakeMembershipRepository(
        {
            (assistant.organization_id, user_id): membership(
                organization_id=assistant.organization_id,
                user_id=user_id,
                role=role,
            )
        }
    )

    actual = await AssistantAccessChecker(
        FakeAssistantRepository({assistant.id: assistant}),
        membership_repository,
    ).require_manager(user_id=user_id, assistant_id=assistant.id)

    assert actual == assistant


@pytest.mark.asyncio
async def test_assistant_manager_access_rejects_member_role() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    membership_repository = FakeMembershipRepository(
        {
            (assistant.organization_id, user_id): membership(
                organization_id=assistant.organization_id,
                user_id=user_id,
                role=OrganizationRole.MEMBER,
            )
        }
    )

    with pytest.raises(
        InsufficientPermissionError,
        match="^Owner or admin role required$",
    ):
        await AssistantAccessChecker(
            FakeAssistantRepository({assistant.id: assistant}),
            membership_repository,
        ).require_manager(user_id=user_id, assistant_id=assistant.id)
