from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from app.application.services import AssistantAccessChecker, OrganizationAccessChecker
from app.application.use_cases import (
    CreateAssistant,
    CreateAssistantCommand,
    DeleteAssistant,
    GetAssistant,
    ListAssistants,
    UpdateAssistant,
    UpdateAssistantCommand,
)
from app.domain.entities import Assistant, OrganizationMember, OrganizationRole
from app.domain.entities.assistant import (
    DEFAULT_ASSISTANT_INSTRUCTIONS,
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_WELCOME_MESSAGE,
)
from app.domain.errors import (
    AssistantNotFoundError,
    InsufficientPermissionError,
    OrganizationNotFoundError,
)


@dataclass
class FakeAssistantRepository:
    assistants: dict[UUID, Assistant] = field(default_factory=dict)
    listed: Sequence[Assistant] = ()
    created: list[Assistant] = field(default_factory=list)
    updated: list[Assistant] = field(default_factory=list)
    deleted: list[UUID] = field(default_factory=list)
    list_calls: list[UUID] = field(default_factory=list)
    delete_result: bool = True

    async def create(self, assistant: Assistant) -> None:
        self.created.append(assistant)

    async def list_by_organization(
        self,
        organization_id: UUID,
    ) -> Sequence[Assistant]:
        self.list_calls.append(organization_id)
        return self.listed

    async def get_by_id(self, assistant_id: UUID) -> Assistant | None:
        return self.assistants.get(assistant_id)

    async def update(self, assistant: Assistant) -> None:
        self.updated.append(assistant)

    async def delete(self, assistant_id: UUID) -> bool:
        self.deleted.append(assistant_id)
        return self.delete_result


@dataclass
class FakeMembershipRepository:
    memberships: dict[tuple[UUID, UUID], OrganizationMember]

    async def get_membership(
        self,
        organization_id: UUID,
        user_id: UUID,
    ) -> OrganizationMember | None:
        return self.memberships.get((organization_id, user_id))


def make_membership(
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


def membership_repository(
    *,
    organization_id: UUID,
    user_id: UUID,
    role: OrganizationRole,
) -> FakeMembershipRepository:
    return FakeMembershipRepository(
        {
            (organization_id, user_id): make_membership(
                organization_id=organization_id,
                user_id=user_id,
                role=role,
            )
        }
    )


def assistant_access_checker(
    assistant_repository: FakeAssistantRepository,
    membership_repository: FakeMembershipRepository,
) -> AssistantAccessChecker:
    return AssistantAccessChecker(
        assistant_repository,
        membership_repository,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [OrganizationRole.OWNER, OrganizationRole.ADMIN],
)
async def test_create_assistant_allows_manager_roles(role: OrganizationRole) -> None:
    organization_id = uuid4()
    user_id = uuid4()
    repository = FakeAssistantRepository()
    use_case = CreateAssistant(
        repository,
        OrganizationAccessChecker(
            membership_repository(
                organization_id=organization_id,
                user_id=user_id,
                role=role,
            )
        ),
    )

    assistant = await use_case.execute(
        CreateAssistantCommand(
            user_id=user_id,
            organization_id=organization_id,
            name="  Support  ",
            description="Customer support assistant",
            welcome_message="Welcome",
            assistant_instructions="Use support documents only.",
            logo_url="https://example.com/logo.png",
            primary_color="#123ABC",
        )
    )

    assert assistant.organization_id == organization_id
    assert assistant.name == "Support"
    assert assistant.description == "Customer support assistant"
    assert assistant.welcome_message == "Welcome"
    assert assistant.assistant_instructions == "Use support documents only."
    assert assistant.logo_url == "https://example.com/logo.png"
    assert assistant.primary_color == "#123ABC"
    assert repository.created == [assistant]


@pytest.mark.asyncio
async def test_create_assistant_applies_product_defaults() -> None:
    organization_id = uuid4()
    user_id = uuid4()
    repository = FakeAssistantRepository()
    use_case = CreateAssistant(
        repository,
        OrganizationAccessChecker(
            membership_repository(
                organization_id=organization_id,
                user_id=user_id,
                role=OrganizationRole.OWNER,
            )
        ),
    )

    assistant = await use_case.execute(
        CreateAssistantCommand(
            user_id=user_id,
            organization_id=organization_id,
            name="Support",
        )
    )

    assert assistant.welcome_message == DEFAULT_WELCOME_MESSAGE
    assert assistant.assistant_instructions == DEFAULT_ASSISTANT_INSTRUCTIONS
    assert assistant.primary_color == DEFAULT_PRIMARY_COLOR


@pytest.mark.asyncio
async def test_create_assistant_rejects_member_role_without_writing() -> None:
    organization_id = uuid4()
    user_id = uuid4()
    repository = FakeAssistantRepository()
    use_case = CreateAssistant(
        repository,
        OrganizationAccessChecker(
            membership_repository(
                organization_id=organization_id,
                user_id=user_id,
                role=OrganizationRole.MEMBER,
            )
        ),
    )

    with pytest.raises(InsufficientPermissionError):
        await use_case.execute(
            CreateAssistantCommand(
                user_id=user_id,
                organization_id=organization_id,
                name="Support",
            )
        )

    assert repository.created == []


@pytest.mark.asyncio
async def test_create_assistant_conceals_cross_tenant_organization() -> None:
    repository = FakeAssistantRepository()
    use_case = CreateAssistant(
        repository,
        OrganizationAccessChecker(FakeMembershipRepository({})),
    )

    with pytest.raises(OrganizationNotFoundError, match="^Organization not found$"):
        await use_case.execute(
            CreateAssistantCommand(
                user_id=uuid4(),
                organization_id=uuid4(),
                name="Support",
            )
        )

    assert repository.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize("role", list(OrganizationRole))
async def test_list_assistants_allows_every_member_role(
    role: OrganizationRole,
) -> None:
    organization_id = uuid4()
    user_id = uuid4()
    expected = (Assistant(organization_id=organization_id, name="Support"),)
    repository = FakeAssistantRepository(listed=expected)
    use_case = ListAssistants(
        repository,
        OrganizationAccessChecker(
            membership_repository(
                organization_id=organization_id,
                user_id=user_id,
                role=role,
            )
        ),
    )

    assistants = await use_case.execute(
        user_id=user_id,
        organization_id=organization_id,
    )

    assert assistants == expected
    assert repository.list_calls == [organization_id]


@pytest.mark.asyncio
async def test_list_assistants_conceals_cross_tenant_organization() -> None:
    repository = FakeAssistantRepository()
    use_case = ListAssistants(
        repository,
        OrganizationAccessChecker(FakeMembershipRepository({})),
    )

    with pytest.raises(OrganizationNotFoundError, match="^Organization not found$"):
        await use_case.execute(user_id=uuid4(), organization_id=uuid4())

    assert repository.list_calls == []


@pytest.mark.asyncio
async def test_get_assistant_returns_assistant_to_member() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    repository = FakeAssistantRepository(assistants={assistant.id: assistant})
    memberships = membership_repository(
        organization_id=assistant.organization_id,
        user_id=user_id,
        role=OrganizationRole.MEMBER,
    )

    actual = await GetAssistant(
        assistant_access_checker(repository, memberships)
    ).execute(user_id=user_id, assistant_id=assistant.id)

    assert actual == assistant


@pytest.mark.asyncio
async def test_get_assistant_conceals_cross_tenant_access() -> None:
    assistant = Assistant(organization_id=uuid4(), name="Private")
    repository = FakeAssistantRepository(assistants={assistant.id: assistant})

    with pytest.raises(AssistantNotFoundError, match="^Assistant not found$"):
        await GetAssistant(
            assistant_access_checker(repository, FakeMembershipRepository({}))
        ).execute(user_id=uuid4(), assistant_id=assistant.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [OrganizationRole.OWNER, OrganizationRole.ADMIN],
)
async def test_update_assistant_allows_manager_roles_and_preserves_ownership(
    role: OrganizationRole,
) -> None:
    user_id = uuid4()
    assistant = Assistant(
        organization_id=uuid4(),
        name="Support",
        description="Old description",
        logo_url="https://example.com/old.png",
    )
    repository = FakeAssistantRepository(assistants={assistant.id: assistant})
    memberships = membership_repository(
        organization_id=assistant.organization_id,
        user_id=user_id,
        role=role,
    )
    use_case = UpdateAssistant(
        repository,
        assistant_access_checker(repository, memberships),
    )

    updated = await use_case.execute(
        UpdateAssistantCommand(
            user_id=user_id,
            assistant_id=assistant.id,
            name="  Help Desk  ",
            description=None,
            assistant_instructions="Use concise answers.",
            logo_url=None,
            primary_color="#ABCDEF",
        )
    )

    assert updated.id == assistant.id
    assert updated.organization_id == assistant.organization_id
    assert updated.created_at == assistant.created_at
    assert updated.name == "Help Desk"
    assert updated.description is None
    assert updated.logo_url is None
    assert updated.primary_color == "#ABCDEF"
    assert updated.welcome_message == assistant.welcome_message
    assert updated.assistant_instructions == "Use concise answers."
    assert updated.updated_at >= assistant.updated_at
    assert repository.updated == [updated]


@pytest.mark.asyncio
async def test_update_assistant_noop_skips_repository_write() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    repository = FakeAssistantRepository(assistants={assistant.id: assistant})
    memberships = membership_repository(
        organization_id=assistant.organization_id,
        user_id=user_id,
        role=OrganizationRole.ADMIN,
    )

    actual = await UpdateAssistant(
        repository,
        assistant_access_checker(repository, memberships),
    ).execute(UpdateAssistantCommand(user_id=user_id, assistant_id=assistant.id))

    assert actual == assistant
    assert repository.updated == []


@pytest.mark.asyncio
async def test_update_assistant_rejects_member_without_writing() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    repository = FakeAssistantRepository(assistants={assistant.id: assistant})
    memberships = membership_repository(
        organization_id=assistant.organization_id,
        user_id=user_id,
        role=OrganizationRole.MEMBER,
    )

    with pytest.raises(InsufficientPermissionError):
        await UpdateAssistant(
            repository,
            assistant_access_checker(repository, memberships),
        ).execute(
            UpdateAssistantCommand(
                user_id=user_id,
                assistant_id=assistant.id,
                name="Blocked",
            )
        )

    assert repository.updated == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [OrganizationRole.OWNER, OrganizationRole.ADMIN],
)
async def test_delete_assistant_allows_manager_roles(role: OrganizationRole) -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    repository = FakeAssistantRepository(assistants={assistant.id: assistant})
    memberships = membership_repository(
        organization_id=assistant.organization_id,
        user_id=user_id,
        role=role,
    )

    await DeleteAssistant(
        repository,
        assistant_access_checker(repository, memberships),
    ).execute(user_id=user_id, assistant_id=assistant.id)

    assert repository.deleted == [assistant.id]


@pytest.mark.asyncio
async def test_delete_assistant_rejects_member_without_deleting() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    repository = FakeAssistantRepository(assistants={assistant.id: assistant})
    memberships = membership_repository(
        organization_id=assistant.organization_id,
        user_id=user_id,
        role=OrganizationRole.MEMBER,
    )

    with pytest.raises(InsufficientPermissionError):
        await DeleteAssistant(
            repository,
            assistant_access_checker(repository, memberships),
        ).execute(user_id=user_id, assistant_id=assistant.id)

    assert repository.deleted == []


@pytest.mark.asyncio
async def test_delete_assistant_handles_record_removed_after_access_check() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    repository = FakeAssistantRepository(
        assistants={assistant.id: assistant},
        delete_result=False,
    )
    memberships = membership_repository(
        organization_id=assistant.organization_id,
        user_id=user_id,
        role=OrganizationRole.OWNER,
    )

    with pytest.raises(AssistantNotFoundError, match="^Assistant not found$"):
        await DeleteAssistant(
            repository,
            assistant_access_checker(repository, memberships),
        ).execute(user_id=user_id, assistant_id=assistant.id)
