from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID

from app.application.services import AssistantAccessChecker, OrganizationAccessChecker
from app.domain.entities import Assistant
from app.domain.entities.assistant import (
    DEFAULT_ASSISTANT_INSTRUCTIONS,
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_WELCOME_MESSAGE,
)
from app.domain.errors import AssistantNotFoundError
from app.domain.ports import AssistantRepository


class _Unset(Enum):
    TOKEN = 0


_UNSET = _Unset.TOKEN


@dataclass(frozen=True, slots=True)
class CreateAssistantCommand:
    user_id: UUID
    organization_id: UUID
    name: str
    description: str | None = None
    welcome_message: str = DEFAULT_WELCOME_MESSAGE
    assistant_instructions: str = DEFAULT_ASSISTANT_INSTRUCTIONS
    logo_url: str | None = None
    primary_color: str = DEFAULT_PRIMARY_COLOR


@dataclass(frozen=True, slots=True)
class UpdateAssistantCommand:
    user_id: UUID
    assistant_id: UUID
    name: str | _Unset = _UNSET
    description: str | None | _Unset = _UNSET
    welcome_message: str | _Unset = _UNSET
    assistant_instructions: str | _Unset = _UNSET
    logo_url: str | None | _Unset = _UNSET
    primary_color: str | _Unset = _UNSET


class CreateAssistant:
    def __init__(
        self,
        repository: AssistantRepository,
        organization_access_checker: OrganizationAccessChecker,
    ) -> None:
        self._repository = repository
        self._organization_access_checker = organization_access_checker

    async def execute(self, command: CreateAssistantCommand) -> Assistant:
        await self._organization_access_checker.require_manager(
            user_id=command.user_id,
            organization_id=command.organization_id,
        )
        assistant = Assistant(
            organization_id=command.organization_id,
            name=command.name.strip(),
            description=command.description,
            welcome_message=command.welcome_message,
            assistant_instructions=command.assistant_instructions,
            logo_url=command.logo_url,
            primary_color=command.primary_color,
        )
        await self._repository.create(assistant)
        return assistant


class ListAssistants:
    def __init__(
        self,
        repository: AssistantRepository,
        organization_access_checker: OrganizationAccessChecker,
    ) -> None:
        self._repository = repository
        self._organization_access_checker = organization_access_checker

    async def execute(
        self,
        *,
        user_id: UUID,
        organization_id: UUID,
    ) -> Sequence[Assistant]:
        await self._organization_access_checker.require_member(
            user_id=user_id,
            organization_id=organization_id,
        )
        return await self._repository.list_by_organization(organization_id)


class GetAssistant:
    def __init__(self, access_checker: AssistantAccessChecker) -> None:
        self._access_checker = access_checker

    async def execute(self, *, user_id: UUID, assistant_id: UUID) -> Assistant:
        return await self._access_checker.require_member(
            user_id=user_id,
            assistant_id=assistant_id,
        )


class UpdateAssistant:
    def __init__(
        self,
        repository: AssistantRepository,
        access_checker: AssistantAccessChecker,
    ) -> None:
        self._repository = repository
        self._access_checker = access_checker

    async def execute(self, command: UpdateAssistantCommand) -> Assistant:
        assistant = await self._access_checker.require_manager(
            user_id=command.user_id,
            assistant_id=command.assistant_id,
        )
        changes = self._changes(command)
        if not changes:
            return assistant

        updated_assistant = replace(
            assistant,
            **changes,
            updated_at=max(datetime.now(UTC), assistant.updated_at),
        )
        await self._repository.update(updated_assistant)
        return updated_assistant

    @staticmethod
    def _changes(command: UpdateAssistantCommand) -> dict[str, object]:
        changes: dict[str, object] = {}
        if command.name is not _UNSET:
            changes["name"] = command.name.strip()
        if command.description is not _UNSET:
            changes["description"] = command.description
        if command.welcome_message is not _UNSET:
            changes["welcome_message"] = command.welcome_message
        if command.assistant_instructions is not _UNSET:
            changes["assistant_instructions"] = command.assistant_instructions
        if command.logo_url is not _UNSET:
            changes["logo_url"] = command.logo_url
        if command.primary_color is not _UNSET:
            changes["primary_color"] = command.primary_color
        return changes


class DeleteAssistant:
    def __init__(
        self,
        repository: AssistantRepository,
        access_checker: AssistantAccessChecker,
    ) -> None:
        self._repository = repository
        self._access_checker = access_checker

    async def execute(self, *, user_id: UUID, assistant_id: UUID) -> None:
        await self._access_checker.require_manager(
            user_id=user_id,
            assistant_id=assistant_id,
        )
        deleted = await self._repository.delete(assistant_id)
        if not deleted:
            raise AssistantNotFoundError("Assistant not found")
