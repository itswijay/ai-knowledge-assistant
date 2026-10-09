from dataclasses import dataclass, field
from uuid import UUID, uuid4

import httpx
import pytest

from app.application.use_cases import (
    AskWidgetQuestionCommand,
    WidgetConfig,
)
from app.dependencies import get_ask_widget_question, get_widget_config
from app.domain.entities import Answer, SourceReference
from app.domain.errors import AssistantNotFoundError
from app.main import create_app


@dataclass
class FakeGetWidgetConfig:
    config: WidgetConfig | None = None
    error: Exception | None = None
    calls: list[UUID] = field(default_factory=list)

    async def execute(self, assistant_id: UUID) -> WidgetConfig:
        self.calls.append(assistant_id)
        if self.error is not None:
            raise self.error
        if self.config is None:
            raise AssertionError("A fake config is required")
        return self.config


@dataclass
class FakeAskWidgetQuestion:
    answer: Answer | None = None
    error: Exception | None = None
    calls: list[AskWidgetQuestionCommand] = field(default_factory=list)

    async def execute(self, command: AskWidgetQuestionCommand) -> Answer:
        self.calls.append(command)
        if self.error is not None:
            raise self.error
        if self.answer is None:
            raise AssertionError("A fake answer is required")
        return self.answer


@pytest.mark.asyncio
async def test_get_widget_config_success_without_auth() -> None:
    assistant_id = uuid4()
    fake_config = WidgetConfig(
        id=assistant_id,
        name="Nihal Fashion Assistant",
        welcome_message="Hello, ask me about clothes!",
        logo_url="https://example.com/logo.png",
        primary_color="#10B981",
    )
    fake_use_case = FakeGetWidgetConfig(config=fake_config)

    application = create_app()
    application.dependency_overrides[get_widget_config] = lambda: fake_use_case

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        response = await client.get(f"/api/v1/widget/{assistant_id}/config")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(assistant_id)
    assert data["name"] == "Nihal Fashion Assistant"
    assert data["welcome_message"] == "Hello, ask me about clothes!"
    assert data["logo_url"] == "https://example.com/logo.png"
    assert data["primary_color"] == "#10B981"
    assert fake_use_case.calls == [assistant_id]


@pytest.mark.asyncio
async def test_get_widget_config_not_found() -> None:
    assistant_id = uuid4()
    fake_use_case = FakeGetWidgetConfig(error=AssistantNotFoundError("Assistant not found"))

    application = create_app()
    application.dependency_overrides[get_widget_config] = lambda: fake_use_case

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        response = await client.get(f"/api/v1/widget/{assistant_id}/config")

    assert response.status_code == 404
    assert response.json()["detail"] == "Assistant not found"


@pytest.mark.asyncio
async def test_post_widget_chat_success_without_auth() -> None:
    assistant_id = uuid4()
    fake_answer = Answer(
        text="Yes, we deliver to Kandy for 350 LKR.",
        sources=(SourceReference(document="delivery_rates.pdf", page=2),),
    )
    fake_use_case = FakeAskWidgetQuestion(answer=fake_answer)

    application = create_app()
    application.dependency_overrides[get_ask_widget_question] = lambda: fake_use_case

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            f"/api/v1/widget/{assistant_id}/chat",
            json={"message": "Do you deliver to Kandy?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Yes, we deliver to Kandy for 350 LKR."
    assert data["sources"] == [{"document": "delivery_rates.pdf", "page": 2}]
    assert len(fake_use_case.calls) == 1
    assert fake_use_case.calls[0].assistant_id == assistant_id
    assert fake_use_case.calls[0].question == "Do you deliver to Kandy?"


@pytest.mark.asyncio
async def test_post_widget_chat_rejects_empty_message() -> None:
    assistant_id = uuid4()
    fake_use_case = FakeAskWidgetQuestion(answer=Answer(text="Fallback"))

    application = create_app()
    application.dependency_overrides[get_ask_widget_question] = lambda: fake_use_case

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            f"/api/v1/widget/{assistant_id}/chat",
            json={"message": "   "},
        )

    assert response.status_code == 422
    assert len(fake_use_case.calls) == 0
