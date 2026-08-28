from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

MAX_ORGANIZATION_NAME_LENGTH = 120


class OrganizationRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


@dataclass(frozen=True, slots=True)
class Organization:
    name: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")
        if len(self.name) > MAX_ORGANIZATION_NAME_LENGTH:
            raise ValueError(
                f"name must not exceed {MAX_ORGANIZATION_NAME_LENGTH} characters"
            )
        _validate_uuid(self.id, field_name="id")
        _validate_timestamps(self.created_at, self.updated_at)


@dataclass(frozen=True, slots=True)
class OrganizationMember:
    organization_id: UUID
    user_id: UUID
    role: OrganizationRole
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _validate_uuid(self.organization_id, field_name="organization_id")
        _validate_uuid(self.user_id, field_name="user_id")
        if not isinstance(self.role, OrganizationRole):
            raise ValueError("role must be an OrganizationRole")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")


def _validate_uuid(value: UUID, *, field_name: str) -> None:
    if value.int == 0:
        raise ValueError(f"{field_name} must not be the nil UUID")


def _validate_timestamps(created_at: datetime, updated_at: datetime) -> None:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    if updated_at.tzinfo is None or updated_at.utcoffset() is None:
        raise ValueError("updated_at must be timezone-aware")
    if updated_at < created_at:
        raise ValueError("updated_at must not precede created_at")
