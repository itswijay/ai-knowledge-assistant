from uuid import UUID

from app.domain.entities.assistant import Assistant
from app.domain.entities.document import Document
from app.domain.entities.organization import OrganizationMember, OrganizationRole
from app.domain.errors import (
    AssistantNotFoundError,
    DocumentNotFoundError,
    InsufficientPermissionError,
    OrganizationNotFoundError,
)
from app.domain.ports.assistant_repository import AssistantRepository
from app.domain.ports.document_repository import DocumentRepository
from app.domain.ports.organization_member_repository import (
    OrganizationMemberRepository,
)

MANAGER_ROLES = frozenset({OrganizationRole.OWNER, OrganizationRole.ADMIN})


class OrganizationAccessChecker:
    def __init__(
        self,
        membership_repository: OrganizationMemberRepository,
    ) -> None:
        self._membership_repository = membership_repository

    async def require_member(
        self,
        *,
        user_id: UUID,
        organization_id: UUID,
    ) -> OrganizationMember:
        membership = await self._membership_repository.get_membership(
            organization_id,
            user_id,
        )
        if membership is None:
            raise OrganizationNotFoundError("Organization not found")
        return membership

    async def require_manager(
        self,
        *,
        user_id: UUID,
        organization_id: UUID,
    ) -> OrganizationMember:
        membership = await self.require_member(
            user_id=user_id,
            organization_id=organization_id,
        )
        if membership.role not in MANAGER_ROLES:
            raise InsufficientPermissionError("Owner or admin role required")
        return membership


class AssistantAccessChecker:
    def __init__(
        self,
        assistant_repository: AssistantRepository,
        membership_repository: OrganizationMemberRepository,
    ) -> None:
        self._assistant_repository = assistant_repository
        self._membership_repository = membership_repository

    async def require_member(
        self,
        *,
        user_id: UUID,
        assistant_id: UUID,
    ) -> Assistant:
        assistant, _ = await self._require_membership(
            user_id=user_id,
            assistant_id=assistant_id,
        )
        return assistant

    async def require_manager(
        self,
        *,
        user_id: UUID,
        assistant_id: UUID,
    ) -> Assistant:
        assistant, membership = await self._require_membership(
            user_id=user_id,
            assistant_id=assistant_id,
        )
        if membership.role not in MANAGER_ROLES:
            raise InsufficientPermissionError("Owner or admin role required")
        return assistant

    async def _require_membership(
        self,
        *,
        user_id: UUID,
        assistant_id: UUID,
    ) -> tuple[Assistant, OrganizationMember]:
        assistant = await self._assistant_repository.get_by_id(assistant_id)
        if assistant is None:
            raise AssistantNotFoundError("Assistant not found")

        membership = await self._membership_repository.get_membership(
            assistant.organization_id,
            user_id,
        )
        if membership is None:
            raise AssistantNotFoundError("Assistant not found")
        return assistant, membership


class DocumentAccessChecker:
    def __init__(
        self,
        document_repository: DocumentRepository,
        assistant_access_checker: AssistantAccessChecker,
    ) -> None:
        self._document_repository = document_repository
        self._assistant_access_checker = assistant_access_checker

    async def require_member(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
    ) -> Document:
        document = await self._get_document(document_id)
        try:
            await self._assistant_access_checker.require_member(
                user_id=user_id,
                assistant_id=document.assistant_id,
            )
        except AssistantNotFoundError as error:
            raise DocumentNotFoundError("Document not found") from error
        return document

    async def require_manager(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
    ) -> Document:
        document = await self._get_document(document_id)
        try:
            await self._assistant_access_checker.require_manager(
                user_id=user_id,
                assistant_id=document.assistant_id,
            )
        except AssistantNotFoundError as error:
            raise DocumentNotFoundError("Document not found") from error
        return document

    async def _get_document(self, document_id: UUID) -> Document:
        document = await self._document_repository.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError("Document not found")
        return document
