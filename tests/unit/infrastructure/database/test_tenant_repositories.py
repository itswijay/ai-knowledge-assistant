from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.domain.entities import (
    Assistant,
    Document,
    Organization,
    OrganizationMember,
    OrganizationRole,
)
from app.domain.errors import (
    AssistantNotFoundError,
    ResourceConflictError,
    TenantRepositoryError,
)
from app.infrastructure.database.assistant_repository import (
    PostgresAssistantRepository,
)
from app.infrastructure.database.document_repository import PostgresDocumentRepository
from app.infrastructure.database.models import (
    AssistantModel,
    DocumentModel,
    OrganizationMemberModel,
    OrganizationModel,
)
from app.infrastructure.database.organization_repository import (
    PostgresOrganizationMemberRepository,
    PostgresOrganizationRepository,
)
from app.infrastructure.database.session import AsyncSessionFactory


class FakeTransaction:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        if exception is None and self._session.commit_error is not None:
            raise self._session.commit_error


@dataclass
class FakeScalarResult:
    models: Sequence[object] = ()

    def scalars(self) -> FakeScalarResult:
        return self

    def all(self) -> Sequence[object]:
        return self.models


@dataclass
class FakeSession:
    result_models: Sequence[object] = ()
    get_result: object | None = None
    operation_error: SQLAlchemyError | None = None
    commit_error: SQLAlchemyError | None = None
    added: list[object] = field(default_factory=list)
    deleted: list[object] = field(default_factory=list)
    get_calls: list[tuple[type[object], object]] = field(default_factory=list)
    executed_statement: object | None = None
    closed: bool = False

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        self.closed = True

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def execute(self, statement: object) -> FakeScalarResult:
        self.executed_statement = statement
        if self.operation_error is not None:
            raise self.operation_error
        return FakeScalarResult(self.result_models)

    async def get(self, model_type: type[object], identity: object) -> object | None:
        self.get_calls.append((model_type, identity))
        if self.operation_error is not None:
            raise self.operation_error
        return self.get_result

    async def delete(self, instance: object) -> None:
        if self.operation_error is not None:
            raise self.operation_error
        self.deleted.append(instance)


@dataclass
class FakeSessionFactory:
    session: FakeSession
    calls: int = 0

    def __call__(self) -> FakeSession:
        self.calls += 1
        return self.session


def session_factory(session: FakeSession) -> AsyncSessionFactory:
    return cast(AsyncSessionFactory, FakeSessionFactory(session))


def organization_model(organization: Organization) -> OrganizationModel:
    return OrganizationModel(
        id=organization.id,
        name=organization.name,
        created_at=organization.created_at,
        updated_at=organization.updated_at,
    )


def membership_model(membership: OrganizationMember) -> OrganizationMemberModel:
    return OrganizationMemberModel(
        organization_id=membership.organization_id,
        user_id=membership.user_id,
        role=membership.role.value,
        created_at=membership.created_at,
    )


def assistant_model(assistant: Assistant) -> AssistantModel:
    return AssistantModel(
        id=assistant.id,
        organization_id=assistant.organization_id,
        name=assistant.name,
        description=assistant.description,
        welcome_message=assistant.welcome_message,
        assistant_instructions=assistant.assistant_instructions,
        logo_url=assistant.logo_url,
        primary_color=assistant.primary_color,
        created_at=assistant.created_at,
        updated_at=assistant.updated_at,
    )


def document_model(document: Document) -> DocumentModel:
    return DocumentModel(
        id=document.id,
        assistant_id=document.assistant_id,
        original_filename=document.original_filename,
        created_at=document.created_at,
    )


@pytest.mark.asyncio
async def test_create_organization_persists_owner_atomically() -> None:
    organization = Organization(name="Example University")
    owner = OrganizationMember(
        organization_id=organization.id,
        user_id=uuid4(),
        role=OrganizationRole.OWNER,
    )
    session = FakeSession()
    repository = PostgresOrganizationRepository(session_factory(session))

    await repository.create_with_owner(organization, owner)

    assert session.closed is True
    assert len(session.added) == 1
    stored = session.added[0]
    assert isinstance(stored, OrganizationModel)
    assert stored.id == organization.id
    assert stored.name == "Example University"
    assert len(stored.members) == 1
    assert stored.members[0].user_id == owner.user_id
    assert stored.members[0].role == "owner"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner",
    [
        OrganizationMember(
            organization_id=uuid4(),
            user_id=uuid4(),
            role=OrganizationRole.OWNER,
        ),
        OrganizationMember(
            organization_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            user_id=uuid4(),
            role=OrganizationRole.ADMIN,
        ),
    ],
)
async def test_create_organization_requires_matching_owner(
    owner: OrganizationMember,
) -> None:
    organization = Organization(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        name="Example",
    )
    session = FakeSession()
    repository = PostgresOrganizationRepository(session_factory(session))

    with pytest.raises(ValueError):
        await repository.create_with_owner(organization, owner)

    assert session.added == []


@pytest.mark.asyncio
async def test_list_organizations_is_user_scoped_in_sql() -> None:
    user_id = uuid4()
    organizations = (
        Organization(name="First"),
        Organization(name="Second"),
    )
    session = FakeSession(
        result_models=[organization_model(item) for item in organizations]
    )
    repository = PostgresOrganizationRepository(session_factory(session))

    result = await repository.list_for_user(user_id)

    assert result == organizations
    statement = session.executed_statement
    assert statement is not None
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "JOIN organization_members" in sql
    assert "organization_members.user_id" in sql
    assert "WHERE" in sql
    assert "ORDER BY" in sql


@pytest.mark.asyncio
async def test_get_organization_maps_model_or_none() -> None:
    organization = Organization(name="Example")
    session = FakeSession(get_result=organization_model(organization))
    repository = PostgresOrganizationRepository(session_factory(session))

    assert await repository.get_by_id(organization.id) == organization
    assert session.get_calls == [(OrganizationModel, organization.id)]

    session.get_result = None
    assert await repository.get_by_id(uuid4()) is None


@pytest.mark.asyncio
async def test_get_membership_uses_composite_identity() -> None:
    membership = OrganizationMember(
        organization_id=uuid4(),
        user_id=uuid4(),
        role=OrganizationRole.ADMIN,
    )
    session = FakeSession(get_result=membership_model(membership))
    repository = PostgresOrganizationMemberRepository(session_factory(session))

    result = await repository.get_membership(
        membership.organization_id,
        membership.user_id,
    )

    assert result == membership
    assert session.get_calls == [
        (
            OrganizationMemberModel,
            (membership.organization_id, membership.user_id),
        )
    ]


@pytest.mark.asyncio
async def test_create_and_list_assistants_preserve_tenant_scope() -> None:
    organization_id = uuid4()
    assistant = Assistant(
        organization_id=organization_id,
        name="Support",
        description="Support assistant",
    )
    create_session = FakeSession()
    repository = PostgresAssistantRepository(session_factory(create_session))

    await repository.create(assistant)

    stored = create_session.added[0]
    assert isinstance(stored, AssistantModel)
    assert stored.organization_id == organization_id
    assert stored.name == "Support"
    assert stored.assistant_instructions == assistant.assistant_instructions

    list_session = FakeSession(result_models=[assistant_model(assistant)])
    repository = PostgresAssistantRepository(session_factory(list_session))
    assert await repository.list_by_organization(organization_id) == (assistant,)
    statement = list_session.executed_statement
    assert statement is not None
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "assistants.organization_id" in sql
    assert "WHERE" in sql
    assert "ORDER BY" in sql


@pytest.mark.asyncio
async def test_get_assistant_maps_model_or_none() -> None:
    assistant = Assistant(organization_id=uuid4(), name="Support")
    session = FakeSession(get_result=assistant_model(assistant))
    repository = PostgresAssistantRepository(session_factory(session))

    assert await repository.get_by_id(assistant.id) == assistant
    session.get_result = None
    assert await repository.get_by_id(uuid4()) is None


@pytest.mark.asyncio
async def test_update_assistant_changes_customization_but_not_ownership() -> None:
    assistant = Assistant(organization_id=uuid4(), name="Support")
    stored = assistant_model(assistant)
    updated = Assistant(
        id=assistant.id,
        organization_id=assistant.organization_id,
        name="Updated Support",
        description="Updated description",
        welcome_message="Welcome",
        assistant_instructions="Use a concise tone.",
        logo_url="https://cdn.example.com/logo.png",
        primary_color="#112233",
        created_at=assistant.created_at,
        updated_at=datetime.now(UTC) + timedelta(seconds=1),
    )
    session = FakeSession(get_result=stored)
    repository = PostgresAssistantRepository(session_factory(session))

    await repository.update(updated)

    assert stored.organization_id == assistant.organization_id
    assert stored.name == "Updated Support"
    assert stored.description == "Updated description"
    assert stored.welcome_message == "Welcome"
    assert stored.assistant_instructions == "Use a concise tone."
    assert stored.logo_url == "https://cdn.example.com/logo.png"
    assert stored.primary_color == "#112233"
    assert stored.created_at == assistant.created_at
    assert stored.updated_at == updated.updated_at


@pytest.mark.asyncio
async def test_update_assistant_rejects_cross_tenant_move() -> None:
    assistant = Assistant(organization_id=uuid4(), name="Support")
    stored = assistant_model(assistant)
    moved = Assistant(
        id=assistant.id,
        organization_id=uuid4(),
        name=assistant.name,
        created_at=assistant.created_at,
        updated_at=assistant.updated_at,
    )
    session = FakeSession(get_result=stored)
    repository = PostgresAssistantRepository(session_factory(session))

    with pytest.raises(ValueError, match="ownership"):
        await repository.update(moved)

    assert stored.organization_id == assistant.organization_id


@pytest.mark.asyncio
async def test_update_missing_assistant_has_structured_error() -> None:
    assistant = Assistant(organization_id=uuid4(), name="Support")
    repository = PostgresAssistantRepository(session_factory(FakeSession()))

    with pytest.raises(AssistantNotFoundError, match="not found"):
        await repository.update(assistant)


@pytest.mark.asyncio
async def test_delete_assistant_reports_whether_row_existed() -> None:
    assistant = Assistant(organization_id=uuid4(), name="Support")
    stored = assistant_model(assistant)
    session = FakeSession(get_result=stored)
    repository = PostgresAssistantRepository(session_factory(session))

    assert await repository.delete(assistant.id) is True
    assert session.deleted == [stored]

    missing_session = FakeSession()
    repository = PostgresAssistantRepository(session_factory(missing_session))
    assert await repository.delete(uuid4()) is False
    assert missing_session.deleted == []


@pytest.mark.asyncio
async def test_list_documents_is_assistant_scoped_in_sql() -> None:
    assistant_id = uuid4()
    documents = (
        Document(assistant_id=assistant_id, original_filename="first.pdf"),
        Document(assistant_id=assistant_id, original_filename="second.pdf"),
    )
    session = FakeSession(
        result_models=[document_model(document) for document in documents]
    )
    repository = PostgresDocumentRepository(session_factory(session))

    assert await repository.list_by_assistant(assistant_id) == documents
    statement = session.executed_statement
    assert statement is not None
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "documents.assistant_id" in sql
    assert "WHERE" in sql
    assert "ORDER BY" in sql
    assert assistant_id in statement.compile().params.values()


@pytest.mark.asyncio
async def test_get_document_maps_model_or_none() -> None:
    document = Document(assistant_id=uuid4(), original_filename="handbook.pdf")
    session = FakeSession(get_result=document_model(document))
    repository = PostgresDocumentRepository(session_factory(session))

    assert await repository.get_by_id(document.id) == document
    assert session.get_calls == [(DocumentModel, document.id)]

    session.get_result = None
    assert await repository.get_by_id(uuid4()) is None


@pytest.mark.asyncio
async def test_delete_document_reports_whether_row_existed() -> None:
    document = Document(assistant_id=uuid4(), original_filename="handbook.pdf")
    stored = document_model(document)
    session = FakeSession(get_result=stored)
    repository = PostgresDocumentRepository(session_factory(session))

    assert await repository.delete(document.id) is True
    assert session.deleted == [stored]

    missing_session = FakeSession()
    repository = PostgresDocumentRepository(session_factory(missing_session))
    assert await repository.delete(uuid4()) is False
    assert missing_session.deleted == []


@pytest.mark.asyncio
async def test_integrity_errors_become_sanitized_conflicts() -> None:
    database_error = IntegrityError(
        "INSERT sensitive SQL",
        {"secret": "value"},
        Exception("constraint details"),
    )
    session = FakeSession(commit_error=database_error)
    repository = PostgresAssistantRepository(session_factory(session))

    with pytest.raises(ResourceConflictError) as error:
        await repository.create(Assistant(organization_id=uuid4(), name="Support"))

    assert "sensitive" not in str(error.value)
    assert "constraint" not in str(error.value)


@pytest.mark.asyncio
async def test_database_errors_are_wrapped_without_query_details() -> None:
    session = FakeSession(operation_error=SQLAlchemyError("sensitive SQL query"))
    repository = PostgresOrganizationRepository(session_factory(session))

    with pytest.raises(TenantRepositoryError) as error:
        await repository.list_for_user(uuid4())

    assert "sensitive" not in str(error.value)
