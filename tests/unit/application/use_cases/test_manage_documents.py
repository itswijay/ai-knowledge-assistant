from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from app.application.use_cases import DeleteDocument, ListDocuments
from app.domain.entities import Assistant, Document
from app.domain.errors import (
    AssistantNotFoundError,
    DocumentNotFoundError,
    InsufficientPermissionError,
)


@dataclass
class FakeDocumentRepository:
    documents: dict[UUID, Document] = field(default_factory=dict)
    listed: Sequence[Document] = ()
    delete_result: bool = True
    list_calls: list[UUID] = field(default_factory=list)
    deleted: list[UUID] = field(default_factory=list)

    async def list_by_assistant(self, assistant_id: UUID) -> Sequence[Document]:
        self.list_calls.append(assistant_id)
        return self.listed

    async def get_by_id(self, document_id: UUID) -> Document | None:
        return self.documents.get(document_id)

    async def delete(self, document_id: UUID) -> bool:
        self.deleted.append(document_id)
        return self.delete_result


@dataclass
class FakeAssistantAccessChecker:
    assistant: Assistant
    error: Exception | None = None
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def require_member(
        self,
        *,
        user_id: UUID,
        assistant_id: UUID,
    ) -> Assistant:
        self.calls.append((user_id, assistant_id))
        if self.error is not None:
            raise self.error
        return self.assistant


@dataclass
class FakeDocumentAccessChecker:
    document: Document
    error: Exception | None = None
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def require_manager(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
    ) -> Document:
        self.calls.append((user_id, document_id))
        if self.error is not None:
            raise self.error
        return self.document


@pytest.mark.asyncio
async def test_list_documents_authorizes_and_uses_assistant_scope() -> None:
    user_id = uuid4()
    assistant = Assistant(organization_id=uuid4(), name="Support")
    documents = (
        Document(assistant_id=assistant.id, original_filename="first.pdf"),
        Document(assistant_id=assistant.id, original_filename="second.pdf"),
    )
    repository = FakeDocumentRepository(listed=documents)
    access_checker = FakeAssistantAccessChecker(assistant)

    actual = await ListDocuments(repository, access_checker).execute(
        user_id=user_id,
        assistant_id=assistant.id,
    )

    assert actual == documents
    assert access_checker.calls == [(user_id, assistant.id)]
    assert repository.list_calls == [assistant.id]


@pytest.mark.asyncio
async def test_list_documents_conceals_cross_tenant_assistant() -> None:
    assistant = Assistant(organization_id=uuid4(), name="Private")
    repository = FakeDocumentRepository()
    access_checker = FakeAssistantAccessChecker(
        assistant,
        error=AssistantNotFoundError("Assistant not found"),
    )

    with pytest.raises(AssistantNotFoundError, match="^Assistant not found$"):
        await ListDocuments(repository, access_checker).execute(
            user_id=uuid4(),
            assistant_id=assistant.id,
        )

    assert repository.list_calls == []


@pytest.mark.asyncio
async def test_delete_document_authorizes_before_deleting() -> None:
    user_id = uuid4()
    document = Document(assistant_id=uuid4(), original_filename="handbook.pdf")
    repository = FakeDocumentRepository(documents={document.id: document})
    access_checker = FakeDocumentAccessChecker(document)

    await DeleteDocument(repository, access_checker).execute(
        user_id=user_id,
        document_id=document.id,
    )

    assert access_checker.calls == [(user_id, document.id)]
    assert repository.deleted == [document.id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        DocumentNotFoundError("Document not found"),
        InsufficientPermissionError("Owner or admin role required"),
    ],
)
async def test_delete_document_denial_stops_repository_write(
    error: Exception,
) -> None:
    document = Document(assistant_id=uuid4(), original_filename="private.pdf")
    repository = FakeDocumentRepository(documents={document.id: document})
    access_checker = FakeDocumentAccessChecker(document, error=error)

    with pytest.raises(type(error), match=str(error)):
        await DeleteDocument(repository, access_checker).execute(
            user_id=uuid4(),
            document_id=document.id,
        )

    assert repository.deleted == []


@pytest.mark.asyncio
async def test_delete_document_handles_record_removed_after_access_check() -> None:
    document = Document(assistant_id=uuid4(), original_filename="handbook.pdf")
    repository = FakeDocumentRepository(
        documents={document.id: document},
        delete_result=False,
    )

    with pytest.raises(DocumentNotFoundError, match="^Document not found$"):
        await DeleteDocument(
            repository,
            FakeDocumentAccessChecker(document),
        ).execute(user_id=uuid4(), document_id=document.id)
