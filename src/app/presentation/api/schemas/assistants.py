from datetime import datetime
from typing import Annotated, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StringConstraints,
    model_validator,
)

from app.domain.entities.assistant import (
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_WELCOME_MESSAGE,
    MAX_ASSISTANT_DESCRIPTION_LENGTH,
    MAX_ASSISTANT_NAME_LENGTH,
    MAX_LOGO_URL_LENGTH,
    MAX_SYSTEM_PROMPT_LENGTH,
    MAX_WELCOME_MESSAGE_LENGTH,
)


def _validate_logo_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("logo_url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("logo_url must not contain credentials")
    return value


AssistantName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_ASSISTANT_NAME_LENGTH,
    ),
]
AssistantDescription = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_ASSISTANT_DESCRIPTION_LENGTH,
    ),
]
WelcomeMessage = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_WELCOME_MESSAGE_LENGTH,
    ),
]
SystemPrompt = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_SYSTEM_PROMPT_LENGTH,
    ),
]
LogoUrl = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_LOGO_URL_LENGTH,
    ),
    AfterValidator(_validate_logo_url),
]
PrimaryColor = Annotated[
    str,
    StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$"),
]


class CreateAssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: AssistantName
    description: AssistantDescription | None = None
    welcome_message: WelcomeMessage = DEFAULT_WELCOME_MESSAGE
    system_prompt: SystemPrompt = DEFAULT_SYSTEM_PROMPT
    logo_url: LogoUrl | None = None
    primary_color: PrimaryColor = DEFAULT_PRIMARY_COLOR


class UpdateAssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: AssistantName | None = None
    description: AssistantDescription | None = None
    welcome_message: WelcomeMessage | None = None
    system_prompt: SystemPrompt | None = None
    logo_url: LogoUrl | None = None
    primary_color: PrimaryColor | None = None

    @model_validator(mode="after")
    def required_fields_must_not_be_null(self) -> Self:
        for field_name in (
            "name",
            "welcome_message",
            "system_prompt",
            "primary_color",
        ):
            if (
                field_name in self.model_fields_set
                and getattr(self, field_name) is None
            ):
                raise ValueError(f"{field_name} must not be null")
        return self


class AssistantResponse(BaseModel):
    id: UUID
    organization_id: UUID
    name: str
    description: str | None
    welcome_message: str
    system_prompt: str
    logo_url: str | None
    primary_color: str
    created_at: datetime
    updated_at: datetime
