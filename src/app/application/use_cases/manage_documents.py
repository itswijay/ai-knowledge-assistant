from collections.abc import Sequence
from uuid import UUID

from app.application.services import AssistantAccessChecker, DocumentAccessChecker
from app.domain.entities import Document
from app.domain.errors import DocumentNotFoundError
from app.domain.ports import DocumentRepository


class ListDocuments:
    def __init__(
        self,
        repository: DocumentRepository,
        assistant_access_checker: AssistantAccessChecker,
    ) -> None:
        self._repository = repository
        self._assistant_access_checker = assistant_access_checker

    async def execute(
        self,
        *,
        user_id: UUID,
        assistant_id: UUID,
    ) -> Sequence[Document]:
        await self._assistant_access_checker.require_member(
            user_id=user_id,
            assistant_id=assistant_id,
        )
        return await self._repository.list_by_assistant(assistant_id)


class DeleteDocument:
    def __init__(
        self,
        repository: DocumentRepository,
        access_checker: DocumentAccessChecker,
    ) -> None:
        self._repository = repository
        self._access_checker = access_checker

    async def execute(self, *, user_id: UUID, document_id: UUID) -> None:
        await self._access_checker.require_manager(
            user_id=user_id,
            document_id=document_id,
        )
        deleted = await self._repository.delete(document_id)
        if not deleted:
            raise DocumentNotFoundError("Document not found")
