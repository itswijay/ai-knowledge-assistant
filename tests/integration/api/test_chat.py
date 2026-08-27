from dataclasses import dataclass, field

import httpx
import pytest

from app.application.constants import FALLBACK_ANSWER
from app.dependencies import get_ask_question
from app.domain.entities import Answer, SourceReference
from app.domain.errors import LLMGenerationError
from app.main import create_app


@dataclass
class FakeAskQuestion:
    answer: Answer | None = None
    error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    async def execute(self, question: str) -> Answer:
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        if self.answer is None:
            raise AssertionError("A fake answer is required")
        return self.answer


async def post_chat(
    fake_use_case: FakeAskQuestion,
    payload: object,
) -> httpx.Response:
    application = create_app()
    application.dependency_overrides[get_ask_question] = lambda: fake_use_case
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post("/api/v1/chat", json=payload)


@pytest.mark.asyncio
async def test_chat_invokes_use_case_and_serializes_grounded_answer() -> None:
    fake_use_case = FakeAskQuestion(
        answer=Answer(
            text="The warranty period is two years.",
            sources=(
                SourceReference(document="warranty.pdf", page=3),
                SourceReference(document="terms.pdf", page=1),
            ),
        )
    )

    response = await post_chat(
        fake_use_case,
        {"message": "  How long is the warranty?  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The warranty period is two years.",
        "sources": [
            {"document": "warranty.pdf", "page": 3},
            {"document": "terms.pdf", "page": 1},
        ],
    }
    assert fake_use_case.calls == ["How long is the warranty?"]


@pytest.mark.asyncio
async def test_chat_serializes_unsupported_answer_without_sources() -> None:
    fake_use_case = FakeAskQuestion(answer=Answer(text=FALLBACK_ANSWER))

    response = await post_chat(
        fake_use_case,
        {"message": "What is the CEO's phone number?"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": FALLBACK_ANSWER, "sources": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": "   "},
        {"message": "x" * 2001},
        {"message": 42},
    ],
)
async def test_chat_rejects_invalid_messages(payload: object) -> None:
    fake_use_case = FakeAskQuestion()

    response = await post_chat(fake_use_case, payload)

    assert response.status_code == 422
    assert fake_use_case.calls == []


@pytest.mark.asyncio
async def test_chat_maps_llm_failure_to_bad_gateway() -> None:
    fake_use_case = FakeAskQuestion(
        error=LLMGenerationError("Gemini answer generation failed")
    )

    response = await post_chat(fake_use_case, {"message": "Warranty?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Gemini answer generation failed"}
