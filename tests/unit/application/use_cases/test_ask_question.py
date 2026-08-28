from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from app.application.constants import FALLBACK_ANSWER
from app.application.services import GroundedPromptBuilder
from app.application.use_cases import AskQuestion, AskQuestionCommand
from app.domain.entities import Assistant, Document, DocumentChunk, RetrievedChunk
from app.domain.errors import AssistantNotFoundError
from app.domain.types import EmbeddingVector


@dataclass
class FakeEmbeddingProvider:
    embedding: EmbeddingVector = (0.1, 0.2)
    query_calls: list[str] = field(default_factory=list)

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        raise AssertionError("Document embedding is not part of question answering")

    async def embed_query(self, text: str) -> EmbeddingVector:
        self.query_calls.append(text)
        return self.embedding


@dataclass
class FakeVectorRepository:
    chunks: Sequence[RetrievedChunk]
    chunks_by_assistant: dict[UUID, Sequence[RetrievedChunk]] | None = None
    search_calls: list[tuple[UUID, EmbeddingVector, int, float]] = field(
        default_factory=list
    )

    async def save_document(
        self,
        document: Document,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        raise AssertionError("Persistence is not part of question answering")

    async def search_similar(
        self,
        assistant_id: UUID,
        query_embedding: EmbeddingVector,
        *,
        limit: int,
        minimum_similarity: float,
    ) -> Sequence[RetrievedChunk]:
        self.search_calls.append(
            (assistant_id, query_embedding, limit, minimum_similarity)
        )
        if self.chunks_by_assistant is not None:
            return self.chunks_by_assistant.get(assistant_id, ())
        return self.chunks


@dataclass
class FakeLLMProvider:
    answer: str = "The warranty period is two years."
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def generate(self, *, system_instruction: str, prompt: str) -> str:
        self.calls.append((system_instruction, prompt))
        return self.answer


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


def retrieved_chunk(
    *,
    chunk_id: int,
    document: str = "warranty.pdf",
    page: int = 3,
    content: str = "The warranty period is two years.",
    similarity: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=chunk_id),
        document_id=UUID(int=100),
        original_filename=document,
        page_number=page,
        chunk_index=chunk_id - 1,
        content=content,
        similarity_score=similarity,
    )


def build_use_case(
    chunks: Sequence[RetrievedChunk],
    *,
    answer: str = "The warranty period is two years.",
    top_k: int = 5,
    similarity_threshold: float = 0.7,
) -> tuple[
    AskQuestion,
    FakeEmbeddingProvider,
    FakeVectorRepository,
    FakeLLMProvider,
    FakeAssistantAccessChecker,
]:
    embedding_provider = FakeEmbeddingProvider()
    repository = FakeVectorRepository(chunks=chunks)
    llm_provider = FakeLLMProvider(answer=answer)
    access_checker = FakeAssistantAccessChecker(
        Assistant(organization_id=uuid4(), name="Support")
    )
    use_case = AskQuestion(
        embedding_provider=embedding_provider,
        vector_repository=repository,
        llm_provider=llm_provider,
        prompt_builder=GroundedPromptBuilder(),
        assistant_access_checker=access_checker,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    return use_case, embedding_provider, repository, llm_provider, access_checker


def command(
    access_checker: FakeAssistantAccessChecker,
    question: str,
    *,
    user_id: UUID | None = None,
) -> AskQuestionCommand:
    return AskQuestionCommand(
        user_id=user_id or uuid4(),
        assistant_id=access_checker.assistant.id,
        question=question,
    )


@pytest.mark.asyncio
async def test_relevant_context_produces_grounded_answer_with_sources() -> None:
    chunks = (
        retrieved_chunk(chunk_id=1),
        retrieved_chunk(
            chunk_id=2,
            page=4,
            content="Claims require the original receipt.",
            similarity=0.82,
        ),
    )
    use_case, embeddings, repository, llm, access_checker = build_use_case(chunks)
    request = command(access_checker, "  How long is the warranty?  ")

    answer = await use_case.execute(request)

    assert access_checker.calls == [(request.user_id, request.assistant_id)]
    assert embeddings.query_calls == ["How long is the warranty?"]
    assert repository.search_calls == [(request.assistant_id, (0.1, 0.2), 5, 0.7)]
    assert answer.text == "The warranty period is two years."
    assert [(source.document, source.page) for source in answer.sources] == [
        ("warranty.pdf", 3),
        ("warranty.pdf", 4),
    ]
    assert len(llm.calls) == 1
    system_instruction, prompt = llm.calls[0]
    assert "using only facts explicitly supported" in system_instruction
    assert "How long is the warranty?" in prompt
    assert "The warranty period is two years." in prompt
    assert "Claims require the original receipt." in prompt


@pytest.mark.asyncio
async def test_duplicate_document_page_sources_are_removed() -> None:
    chunks = (
        retrieved_chunk(chunk_id=1),
        retrieved_chunk(chunk_id=2, content="Coverage begins on purchase."),
    )
    use_case, _, _, _, access_checker = build_use_case(chunks)

    answer = await use_case.execute(
        command(access_checker, "How long is the warranty?")
    )

    assert [(source.document, source.page) for source in answer.sources] == [
        ("warranty.pdf", 3)
    ]


@pytest.mark.asyncio
async def test_missing_context_returns_fallback_without_calling_llm() -> None:
    use_case, _, _, llm, access_checker = build_use_case([])

    answer = await use_case.execute(
        command(access_checker, "What is the return window?")
    )

    assert answer.text == FALLBACK_ANSWER
    assert answer.sources == ()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_low_confidence_results_return_fallback_without_calling_llm() -> None:
    use_case, _, repository, llm, access_checker = build_use_case(
        [retrieved_chunk(chunk_id=1, similarity=0.69)]
    )

    request = command(access_checker, "What is the return window?")
    answer = await use_case.execute(request)

    assert repository.search_calls == [(request.assistant_id, (0.1, 0.2), 5, 0.7)]
    assert answer.text == FALLBACK_ANSWER
    assert answer.sources == ()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_evaluation_trace_retains_low_confidence_retrieval() -> None:
    low_confidence_chunk = retrieved_chunk(chunk_id=1, similarity=0.3)
    use_case, _, repository, llm, access_checker = build_use_case(
        [low_confidence_chunk]
    )
    request = command(access_checker, "What is the retail price?")

    trace = await use_case.execute_with_trace(request)

    assert repository.search_calls == [(request.assistant_id, (0.1, 0.2), 5, 0.0)]
    assert trace.answer.text == FALLBACK_ANSWER
    assert trace.answer.sources == ()
    assert trace.retrieved_chunks == (low_confidence_chunk,)
    assert llm.calls == []


@pytest.mark.asyncio
async def test_only_sufficient_chunks_are_sent_to_llm_and_exposed_as_sources() -> None:
    chunks = (
        retrieved_chunk(chunk_id=1, similarity=0.91),
        retrieved_chunk(
            chunk_id=2,
            document="unrelated.pdf",
            page=7,
            content="This result is below the threshold.",
            similarity=0.2,
        ),
    )
    use_case, _, _, llm, access_checker = build_use_case(chunks)

    answer = await use_case.execute(
        command(access_checker, "How long is the warranty?")
    )

    assert [(source.document, source.page) for source in answer.sources] == [
        ("warranty.pdf", 3)
    ]
    assert "This result is below the threshold." not in llm.calls[0][1]


@pytest.mark.asyncio
async def test_llm_fallback_is_returned_without_sources() -> None:
    use_case, _, _, llm, access_checker = build_use_case(
        [retrieved_chunk(chunk_id=1)],
        answer=f"  {FALLBACK_ANSWER}  ",
    )

    answer = await use_case.execute(
        command(access_checker, "Is accidental damage covered?")
    )

    assert answer.text == FALLBACK_ANSWER
    assert answer.sources == ()
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_blank_question_is_rejected_before_external_calls() -> None:
    use_case, embeddings, repository, llm, access_checker = build_use_case([])

    with pytest.raises(ValueError, match="question must not be blank"):
        await use_case.execute(command(access_checker, "   "))

    assert embeddings.query_calls == []
    assert repository.search_calls == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_cross_tenant_chunks_are_never_returned_for_target_assistant() -> None:
    assistant_a_chunk = retrieved_chunk(
        chunk_id=1,
        document="assistant-a.pdf",
        similarity=0.8,
    )
    assistant_b_chunk = retrieved_chunk(
        chunk_id=2,
        document="assistant-b-private.pdf",
        similarity=0.99,
    )
    use_case, _, repository, _, access_checker = build_use_case([])
    assistant_a_id = access_checker.assistant.id
    assistant_b_id = uuid4()
    repository.chunks_by_assistant = {
        assistant_a_id: (assistant_a_chunk,),
        assistant_b_id: (assistant_b_chunk,),
    }

    answer = await use_case.execute(
        AskQuestionCommand(
            user_id=uuid4(),
            assistant_id=assistant_a_id,
            question="What does my document say?",
        )
    )

    assert [(source.document, source.page) for source in answer.sources] == [
        ("assistant-a.pdf", 3)
    ]
    assert assistant_b_chunk not in repository.chunks_by_assistant[assistant_a_id]
    assert repository.search_calls[0][0] == assistant_a_id


@pytest.mark.asyncio
async def test_cross_tenant_access_stops_before_embedding_or_retrieval() -> None:
    use_case, embeddings, repository, llm, access_checker = build_use_case([])
    access_checker.error = AssistantNotFoundError("Assistant not found")

    with pytest.raises(AssistantNotFoundError, match="^Assistant not found$"):
        await use_case.execute(command(access_checker, "Private question"))

    assert embeddings.query_calls == []
    assert repository.search_calls == []
    assert llm.calls == []


@pytest.mark.parametrize(
    ("top_k", "similarity_threshold", "message"),
    [
        (0, 0.7, "top_k"),
        (51, 0.7, "top_k"),
        (5, -0.1, "similarity_threshold"),
        (5, 1.1, "similarity_threshold"),
        (5, float("nan"), "similarity_threshold"),
    ],
)
def test_invalid_retrieval_configuration_is_rejected(
    top_k: int,
    similarity_threshold: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AskQuestion(
            embedding_provider=FakeEmbeddingProvider(),
            vector_repository=FakeVectorRepository(chunks=[]),
            llm_provider=FakeLLMProvider(),
            prompt_builder=GroundedPromptBuilder(),
            assistant_access_checker=FakeAssistantAccessChecker(
                Assistant(organization_id=uuid4(), name="Support")
            ),
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
