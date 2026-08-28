from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.domain.entities import (
    Assistant,
    Organization,
    OrganizationMember,
    OrganizationRole,
)
from app.domain.entities.assistant import (
    DEFAULT_ASSISTANT_INSTRUCTIONS,
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_WELCOME_MESSAGE,
)
from app.domain.errors import (
    AssistantNotFoundError,
    AuthorizationError,
    DocumentNotFoundError,
    InsufficientPermissionError,
    MembershipRequiredError,
    OrganizationNotFoundError,
    ResourceConflictError,
    ResourceNotFoundError,
)


def test_organization_has_uuid_and_utc_timestamps() -> None:
    organization = Organization(name="Example University")

    assert organization.id.int != 0
    assert organization.created_at.tzinfo is UTC
    assert organization.updated_at.tzinfo is UTC


@pytest.mark.parametrize("name", ["", "   ", "x" * 121])
def test_organization_rejects_invalid_name(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        Organization(name=name)


def test_organization_rejects_invalid_identity_and_timestamps() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValueError, match="nil UUID"):
        Organization(name="Example", id=UUID(int=0))
    with pytest.raises(ValueError, match="must not precede"):
        Organization(
            name="Example",
            created_at=now,
            updated_at=now - timedelta(seconds=1),
        )


def test_organization_member_uses_supported_role() -> None:
    membership = OrganizationMember(
        organization_id=uuid4(),
        user_id=uuid4(),
        role=OrganizationRole.OWNER,
    )

    assert membership.role is OrganizationRole.OWNER
    assert {role.value for role in OrganizationRole} == {"owner", "admin", "member"}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OrganizationMember(
            organization_id=UUID(int=0),
            user_id=uuid4(),
            role=OrganizationRole.MEMBER,
        ),
        lambda: OrganizationMember(
            organization_id=uuid4(),
            user_id=UUID(int=0),
            role=OrganizationRole.MEMBER,
        ),
        lambda: OrganizationMember(
            organization_id=uuid4(),
            user_id=uuid4(),
            role="owner",  # type: ignore[arg-type]
        ),
    ],
)
def test_organization_member_rejects_invalid_values(
    factory: Callable[[], OrganizationMember],
) -> None:
    with pytest.raises(ValueError):
        factory()


def test_assistant_has_safe_customization_defaults() -> None:
    organization_id = uuid4()

    assistant = Assistant(organization_id=organization_id, name="Student Support")

    assert assistant.organization_id == organization_id
    assert assistant.welcome_message == DEFAULT_WELCOME_MESSAGE
    assert assistant.assistant_instructions == DEFAULT_ASSISTANT_INSTRUCTIONS
    assert assistant.primary_color == DEFAULT_PRIMARY_COLOR
    assert assistant.description is None
    assert assistant.logo_url is None


def test_assistant_accepts_bounded_customization() -> None:
    assistant = Assistant(
        organization_id=uuid4(),
        name="Support",
        description="Answers product support questions.",
        welcome_message="Welcome to support.",
        assistant_instructions="Use a concise and professional tone.",
        logo_url="https://cdn.example.com/assistant/logo.png?v=2",
        primary_color="#a1B2c3",
    )

    assert assistant.logo_url == "https://cdn.example.com/assistant/logo.png?v=2"
    assert assistant.primary_color == "#a1B2c3"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Assistant(organization_id=UUID(int=0), name="Support"),
        lambda: Assistant(organization_id=uuid4(), name=""),
        lambda: Assistant(organization_id=uuid4(), name="x" * 101),
        lambda: Assistant(organization_id=uuid4(), name="Support", description=" "),
        lambda: Assistant(
            organization_id=uuid4(),
            name="Support",
            welcome_message="x" * 501,
        ),
        lambda: Assistant(
            organization_id=uuid4(),
            name="Support",
            assistant_instructions=" ",
        ),
        lambda: Assistant(
            organization_id=uuid4(),
            name="Support",
            logo_url="javascript:alert(1)",
        ),
        lambda: Assistant(
            organization_id=uuid4(),
            name="Support",
            logo_url="https://user:password@cdn.example.com/logo.png",
        ),
        lambda: Assistant(
            organization_id=uuid4(),
            name="Support",
            primary_color="blue",
        ),
    ],
)
def test_assistant_rejects_invalid_values(factory: Callable[[], Assistant]) -> None:
    with pytest.raises(ValueError):
        factory()


def test_tenant_entities_are_immutable() -> None:
    organization = Organization(name="Example")
    membership = OrganizationMember(
        organization_id=organization.id,
        user_id=uuid4(),
        role=OrganizationRole.MEMBER,
    )
    assistant = Assistant(organization_id=organization.id, name="Support")

    with pytest.raises(FrozenInstanceError):
        organization.name = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        membership.role = OrganizationRole.ADMIN  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        assistant.name = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "error_type",
    [OrganizationNotFoundError, AssistantNotFoundError, DocumentNotFoundError],
)
def test_resource_errors_share_not_found_base(
    error_type: type[ResourceNotFoundError],
) -> None:
    assert isinstance(error_type("Not found"), ResourceNotFoundError)


@pytest.mark.parametrize(
    "error_type",
    [MembershipRequiredError, InsufficientPermissionError],
)
def test_access_errors_share_authorization_base(
    error_type: type[AuthorizationError],
) -> None:
    assert isinstance(error_type("Access denied"), AuthorizationError)


def test_resource_conflict_is_not_an_authorization_error() -> None:
    assert not isinstance(ResourceConflictError("Conflict"), AuthorizationError)
