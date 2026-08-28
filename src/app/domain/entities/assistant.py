import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import UUID, uuid4

DEFAULT_WELCOME_MESSAGE = "Hi! How can I help you today?"
DEFAULT_ASSISTANT_INSTRUCTIONS = "Answer questions using the provided knowledge base."
DEFAULT_PRIMARY_COLOR = "#2563EB"
MAX_ASSISTANT_NAME_LENGTH = 100
MAX_ASSISTANT_DESCRIPTION_LENGTH = 1000
MAX_WELCOME_MESSAGE_LENGTH = 500
MAX_ASSISTANT_INSTRUCTIONS_LENGTH = 4000
MAX_LOGO_URL_LENGTH = 2048
HEX_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}\Z")


@dataclass(frozen=True, slots=True)
class Assistant:
    organization_id: UUID
    name: str
    description: str | None = None
    welcome_message: str = DEFAULT_WELCOME_MESSAGE
    assistant_instructions: str = DEFAULT_ASSISTANT_INSTRUCTIONS
    logo_url: str | None = None
    primary_color: str = DEFAULT_PRIMARY_COLOR
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _validate_uuid(self.organization_id, field_name="organization_id")
        _validate_uuid(self.id, field_name="id")
        _validate_required_text(
            self.name,
            field_name="name",
            maximum_length=MAX_ASSISTANT_NAME_LENGTH,
        )
        _validate_optional_text(
            self.description,
            field_name="description",
            maximum_length=MAX_ASSISTANT_DESCRIPTION_LENGTH,
        )
        _validate_required_text(
            self.welcome_message,
            field_name="welcome_message",
            maximum_length=MAX_WELCOME_MESSAGE_LENGTH,
        )
        _validate_required_text(
            self.assistant_instructions,
            field_name="assistant_instructions",
            maximum_length=MAX_ASSISTANT_INSTRUCTIONS_LENGTH,
        )
        _validate_logo_url(self.logo_url)
        if HEX_COLOR_PATTERN.fullmatch(self.primary_color) is None:
            raise ValueError("primary_color must use the #RRGGBB format")
        _validate_timestamps(self.created_at, self.updated_at)


def _validate_uuid(value: UUID, *, field_name: str) -> None:
    if value.int == 0:
        raise ValueError(f"{field_name} must not be the nil UUID")


def _validate_required_text(
    value: str,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if len(value) > maximum_length:
        raise ValueError(f"{field_name} must not exceed {maximum_length} characters")


def _validate_optional_text(
    value: str | None,
    *,
    field_name: str,
    maximum_length: int,
) -> None:
    if value is None:
        return
    _validate_required_text(
        value,
        field_name=field_name,
        maximum_length=maximum_length,
    )


def _validate_logo_url(value: str | None) -> None:
    if value is None:
        return
    _validate_required_text(
        value,
        field_name="logo_url",
        maximum_length=MAX_LOGO_URL_LENGTH,
    )
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("logo_url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("logo_url must not contain credentials")


def _validate_timestamps(created_at: datetime, updated_at: datetime) -> None:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    if updated_at.tzinfo is None or updated_at.utcoffset() is None:
        raise ValueError("updated_at must be timezone-aware")
    if updated_at < created_at:
        raise ValueError("updated_at must not precede created_at")
