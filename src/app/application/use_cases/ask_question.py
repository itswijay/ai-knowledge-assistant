from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from app.application.constants import FALLBACK_ANSWER
from app.application.services import AssistantAccessChecker, GroundedPromptBuilder
from app.domain.entities import Answer, RetrievedChunk, SourceReference
from app.domain.ports import EmbeddingProvider, LLMProvider, VectorRepository


@dataclass(frozen=True, slots=True)
class QuestionAnswerTrace:
    answer: Answer
    retrieved_chunks: tuple[RetrievedChunk, ...]


@dataclass(frozen=True, slots=True)
class AskQuestionCommand:
    user_id: UUID
    assistant_id: UUID
    question: str


class AskQuestion:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_repository: VectorRepository,
        llm_provider: LLMProvider,
        prompt_builder: GroundedPromptBuilder,
        assistant_access_checker: AssistantAccessChecker,
        top_k: int,
        similarity_threshold: float,
    ) -> None:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not isfinite(similarity_threshold) or not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self._embedding_provider = embedding_provider
        self._vector_repository = vector_repository
        self._llm_provider = llm_provider
        self._prompt_builder = prompt_builder
        self._assistant_access_checker = assistant_access_checker
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    async def execute(self, command: AskQuestionCommand) -> Answer:
        trace = await self._execute(
            command,
            retrieval_minimum_similarity=self._similarity_threshold,
        )
        return trace.answer

    async def execute_with_trace(
        self,
        command: AskQuestionCommand,
    ) -> QuestionAnswerTrace:
        """Answer while retaining low-confidence retrieval data for evaluation."""

        return await self._execute(command, retrieval_minimum_similarity=0.0)

    async def _execute(
        self,
        command: AskQuestionCommand,
        *,
        retrieval_minimum_similarity: float,
    ) -> QuestionAnswerTrace:
        assistant = await self._assistant_access_checker.require_member(
            user_id=command.user_id,
            assistant_id=command.assistant_id,
        )
        cleaned_question = command.question.strip()
        if not cleaned_question:
            raise ValueError("question must not be blank")

        query_embedding = await self._embedding_provider.embed_query(cleaned_question)
        retrieved_chunks = await self._vector_repository.search_similar(
            command.assistant_id,
            query_embedding,
            limit=self._top_k,
            minimum_similarity=retrieval_minimum_similarity,
        )
        traced_chunks = tuple(retrieved_chunks[: self._top_k])
        sufficient_chunks = self._select_sufficient_chunks(traced_chunks)
        if not sufficient_chunks:
            return QuestionAnswerTrace(
                answer=self._fallback(),
                retrieved_chunks=traced_chunks,
            )

        grounded_prompt = self._prompt_builder.build(
            question=cleaned_question,
            chunks=sufficient_chunks,
            assistant_instructions=assistant.system_prompt,
        )
        answer_text = (
            await self._llm_provider.generate(
                system_instruction=grounded_prompt.system_instruction,
                prompt=grounded_prompt.prompt,
            )
        ).strip()
        if answer_text == FALLBACK_ANSWER:
            return QuestionAnswerTrace(
                answer=self._fallback(),
                retrieved_chunks=traced_chunks,
            )

        return QuestionAnswerTrace(
            answer=Answer(
                text=answer_text,
                sources=self._collect_sources(sufficient_chunks),
            ),
            retrieved_chunks=traced_chunks,
        )

    def _select_sufficient_chunks(
        self,
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[RetrievedChunk, ...]:
        return tuple(
            chunk
            for chunk in chunks[: self._top_k]
            if chunk.similarity_score >= self._similarity_threshold
        )

    @staticmethod
    def _collect_sources(
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[SourceReference, ...]:
        unique_sources: dict[tuple[str, int], SourceReference] = {}
        for chunk in chunks:
            key = (chunk.original_filename, chunk.page_number)
            unique_sources.setdefault(
                key,
                SourceReference(
                    document=chunk.original_filename, page=chunk.page_number
                ),
            )
        return tuple(unique_sources.values())

    @staticmethod
    def _fallback() -> Answer:
        return Answer(text=FALLBACK_ANSWER)
