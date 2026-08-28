from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from app.application.use_cases import (
    CreateAssistantCommand,
    UpdateAssistantCommand,
)
from app.dependencies import (
    get_access_token_verifier,
    get_create_assistant,
    get_delete_assistant,
    get_get_assistant,
    get_list_assistants,
    get_update_assistant,
)
from app.domain.entities import Assistant, AuthenticatedUser
from app.domain.entities.assistant import (
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_WELCOME_MESSAGE,
)
from app.domain.errors import AssistantNotFoundError, InsufficientPermissionError
from app.main import create_app


@dataclass
class FakeAccessTokenVerifier:
    user: AuthenticatedUser
    tokens: list[str] = field(default_factory=list)

    async def verify(self, token: str) -> AuthenticatedUser:
        self.tokens.append(token)
        return self.user


@dataclass
class FakeCreateAssistant:
    assistant: Assistant | None = None
    error: Exception | None = None
    calls: list[CreateAssistantCommand] = field(default_factory=list)

    async def execute(self, command: CreateAssistantCommand) -> Assistant:
        self.calls.append(command)
        if self.error is not None:
            raise self.error
        if self.assistant is None:
            raise AssertionError("A fake assistant is required")
        return self.assistant


@dataclass
class FakeListAssistants:
    assistants: Sequence[Assistant] = ()
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def execute(
        self,
        *,
        user_id: UUID,
        organization_id: UUID,
    ) -> Sequence[Assistant]:
        self.calls.append((user_id, organization_id))
        return self.assistants


@dataclass
class FakeGetAssistant:
    assistant: Assistant | None = None
    error: Exception | None = None
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def execute(self, *, user_id: UUID, assistant_id: UUID) -> Assistant:
        self.calls.append((user_id, assistant_id))
        if self.error is not None:
            raise self.error
        if self.assistant is None:
            raise AssertionError("A fake assistant is required")
        return self.assistant


@dataclass
class FakeUpdateAssistant:
    assistant: Assistant | None = None
    error: Exception | None = None
    calls: list[UpdateAssistantCommand] = field(default_factory=list)

    async def execute(self, command: UpdateAssistantCommand) -> Assistant:
        self.calls.append(command)
        if self.error is not None:
            raise self.error
        if self.assistant is None:
            raise AssertionError("A fake assistant is required")
        return self.assistant


@dataclass
class FakeDeleteAssistant:
    error: Exception | None = None
    calls: list[tuple[UUID, UUID]] = field(default_factory=list)

    async def execute(self, *, user_id: UUID, assistant_id: UUID) -> None:
        self.calls.append((user_id, assistant_id))
        if self.error is not None:
            raise self.error


async def request_assistant_endpoint(
    *,
    method: str,
    path: str,
    dependency: Callable[..., Any],
    use_case: object,
    user: AuthenticatedUser,
    json: object | None = None,
    authenticated: bool = True,
) -> tuple[httpx.Response, FakeAccessTokenVerifier]:
    application = create_app()
    verifier = FakeAccessTokenVerifier(user)

    async def override_verifier() -> FakeAccessTokenVerifier:
        return verifier

    async def override_use_case() -> object:
        return use_case

    application.dependency_overrides[get_access_token_verifier] = override_verifier
    application.dependency_overrides[dependency] = override_use_case
    headers = {"Authorization": "Bearer valid-token"} if authenticated else {}
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.request(
            method,
            path,
            headers=headers,
            json=json,
        )
    return response, verifier


@pytest.mark.asyncio
async def test_create_assistant_uses_verified_user_and_path_organization() -> None:
    user = AuthenticatedUser(id=uuid4())
    organization_id = uuid4()
    assistant = Assistant(
        organization_id=organization_id,
        name="Support",
        assistant_instructions="Be concise and friendly.",
    )
    use_case = FakeCreateAssistant(assistant=assistant)

    response, verifier = await request_assistant_endpoint(
        method="POST",
        path=f"/api/v1/organizations/{organization_id}/assistants",
        dependency=get_create_assistant,
        use_case=use_case,
        user=user,
        json={
            "name": "  Support  ",
            "assistant_instructions": "Be concise and friendly.",
        },
    )

    assert response.status_code == 201
    assert response.json()["id"] == str(assistant.id)
    assert response.json()["organization_id"] == str(organization_id)
    assert response.json()["name"] == "Support"
    assert response.json()["assistant_instructions"] == "Be concise and friendly."
    assert "system_prompt" not in response.json()
    assert use_case.calls == [
        CreateAssistantCommand(
            user_id=user.id,
            organization_id=organization_id,
            name="Support",
            description=None,
            welcome_message=DEFAULT_WELCOME_MESSAGE,
            assistant_instructions="Be concise and friendly.",
            logo_url=None,
            primary_color=DEFAULT_PRIMARY_COLOR,
        )
    ]
    assert verifier.tokens == ["valid-token"]


@pytest.mark.asyncio
async def test_list_assistants_uses_verified_user_and_path_organization() -> None:
    user = AuthenticatedUser(id=uuid4())
    organization_id = uuid4()
    assistants = (
        Assistant(organization_id=organization_id, name="Support"),
        Assistant(organization_id=organization_id, name="Sales"),
    )
    use_case = FakeListAssistants(assistants=assistants)

    response, _ = await request_assistant_endpoint(
        method="GET",
        path=f"/api/v1/organizations/{organization_id}/assistants",
        dependency=get_list_assistants,
        use_case=use_case,
        user=user,
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [
        str(assistant.id) for assistant in assistants
    ]
    assert use_case.calls == [(user.id, organization_id)]


@pytest.mark.asyncio
async def test_get_assistant_uses_verified_user_and_path_id() -> None:
    user = AuthenticatedUser(id=uuid4())
    assistant = Assistant(organization_id=uuid4(), name="Support")
    use_case = FakeGetAssistant(assistant=assistant)

    response, _ = await request_assistant_endpoint(
        method="GET",
        path=f"/api/v1/assistants/{assistant.id}",
        dependency=get_get_assistant,
        use_case=use_case,
        user=user,
    )

    assert response.status_code == 200
    assert response.json()["id"] == str(assistant.id)
    assert use_case.calls == [(user.id, assistant.id)]


@pytest.mark.asyncio
async def test_patch_preserves_omitted_fields_and_allows_optional_nulls() -> None:
    user = AuthenticatedUser(id=uuid4())
    assistant = Assistant(
        organization_id=uuid4(),
        name="Support",
        description=None,
        assistant_instructions="Use concise answers.",
        logo_url=None,
    )
    use_case = FakeUpdateAssistant(assistant=assistant)

    response, _ = await request_assistant_endpoint(
        method="PATCH",
        path=f"/api/v1/assistants/{assistant.id}",
        dependency=get_update_assistant,
        use_case=use_case,
        user=user,
        json={
            "description": None,
            "assistant_instructions": "Use concise answers.",
            "logo_url": None,
        },
    )

    assert response.status_code == 200
    command = use_case.calls[0]
    empty_command = UpdateAssistantCommand(
        user_id=user.id,
        assistant_id=assistant.id,
    )
    assert command.user_id == user.id
    assert command.assistant_id == assistant.id
    assert command.description is None
    assert command.assistant_instructions == "Use concise answers."
    assert command.logo_url is None
    assert command.name == empty_command.name
    assert command.welcome_message == empty_command.welcome_message
    assert empty_command.assistant_instructions != command.assistant_instructions
    assert command.primary_color == empty_command.primary_color


@pytest.mark.asyncio
async def test_delete_assistant_returns_no_content() -> None:
    user = AuthenticatedUser(id=uuid4())
    assistant_id = uuid4()
    use_case = FakeDeleteAssistant()

    response, _ = await request_assistant_endpoint(
        method="DELETE",
        path=f"/api/v1/assistants/{assistant_id}",
        dependency=get_delete_assistant,
        use_case=use_case,
        user=user,
    )

    assert response.status_code == 204
    assert response.content == b""
    assert use_case.calls == [(user.id, assistant_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"name": "   "},
        {"name": "x" * 101},
        {"name": "Support", "primary_color": "blue"},
        {"name": "Support", "logo_url": "relative/logo.png"},
        {"name": "Support", "logo_url": "https://user:pass@example.com/logo"},
    ],
)
async def test_create_assistant_rejects_invalid_payload(payload: object) -> None:
    use_case = FakeCreateAssistant()

    response, _ = await request_assistant_endpoint(
        method="POST",
        path=f"/api/v1/organizations/{uuid4()}/assistants",
        dependency=get_create_assistant,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        json=payload,
    )

    assert response.status_code == 422
    assert use_case.calls == []


@pytest.mark.asyncio
async def test_create_assistant_rejects_client_supplied_tenant_identity() -> None:
    use_case = FakeCreateAssistant()

    response, _ = await request_assistant_endpoint(
        method="POST",
        path=f"/api/v1/organizations/{uuid4()}/assistants",
        dependency=get_create_assistant,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        json={"name": "Support", "user_id": str(uuid4())},
    )

    assert response.status_code == 422
    assert use_case.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"name": None},
        {"welcome_message": None},
        {"assistant_instructions": None},
        {"primary_color": None},
    ],
)
async def test_patch_rejects_null_for_required_fields(payload: object) -> None:
    use_case = FakeUpdateAssistant()

    response, _ = await request_assistant_endpoint(
        method="PATCH",
        path=f"/api/v1/assistants/{uuid4()}",
        dependency=get_update_assistant,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        json=payload,
    )

    assert response.status_code == 422
    assert use_case.calls == []


@pytest.mark.asyncio
async def test_assistant_endpoints_require_authentication() -> None:
    use_case = FakeGetAssistant()

    response, verifier = await request_assistant_endpoint(
        method="GET",
        path=f"/api/v1/assistants/{uuid4()}",
        dependency=get_get_assistant,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        authenticated=False,
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication credentials were not provided"}
    assert verifier.tokens == []
    assert use_case.calls == []


@pytest.mark.asyncio
async def test_cross_tenant_assistant_is_concealed_as_not_found() -> None:
    use_case = FakeGetAssistant(error=AssistantNotFoundError("Assistant not found"))

    response, _ = await request_assistant_endpoint(
        method="GET",
        path=f"/api/v1/assistants/{uuid4()}",
        dependency=get_get_assistant,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Assistant not found"}


@pytest.mark.asyncio
async def test_member_write_is_forbidden() -> None:
    use_case = FakeUpdateAssistant(
        error=InsufficientPermissionError("Owner or admin role required")
    )

    response, _ = await request_assistant_endpoint(
        method="PATCH",
        path=f"/api/v1/assistants/{uuid4()}",
        dependency=get_update_assistant,
        use_case=use_case,
        user=AuthenticatedUser(id=uuid4()),
        json={"name": "Blocked"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Owner or admin role required"}


@pytest.mark.asyncio
async def test_openapi_documents_assistant_bearer_authentication() -> None:
    application = create_app()
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/openapi.json")

    paths = response.json()["paths"]
    collection = paths["/api/v1/organizations/{organization_id}/assistants"]
    item = paths["/api/v1/assistants/{assistant_id}"]
    assert collection["post"]["security"] == [{"HTTPBearer": []}]
    assert collection["get"]["security"] == [{"HTTPBearer": []}]
    assert item["get"]["security"] == [{"HTTPBearer": []}]
    assert item["patch"]["security"] == [{"HTTPBearer": []}]
    assert item["delete"]["security"] == [{"HTTPBearer": []}]
