from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from app.application.constants import FALLBACK_ANSWER
from app.application.services import (
    GroundedPromptBuilder,
    build_greeting_prompt,
    is_conversational_greeting,
)
from app.domain.entities import Answer, RetrievedChunk, SourceReference
from app.domain.errors import AssistantNotFoundError
from app.domain.ports import (
    AssistantRepository,
    EmbeddingProvider,
    LLMProvider,
    VectorRepository,
)


@dataclass(frozen=True, slots=True)
class WidgetConfig:
    id: UUID
    name: str
    welcome_message: str
    logo_url: str | None
    primary_color: str


@dataclass(frozen=True, slots=True)
class AskWidgetQuestionCommand:
    assistant_id: UUID
    question: str


class GetWidgetConfig:
    def __init__(self, assistant_repository: AssistantRepository) -> None:
        self._assistant_repository = assistant_repository

    async def execute(self, assistant_id: UUID) -> WidgetConfig:
        assistant = await self._assistant_repository.get_by_id(assistant_id)
        if assistant is None:
            raise AssistantNotFoundError("Assistant not found")
        return WidgetConfig(
            id=assistant.id,
            name=assistant.name,
            welcome_message=assistant.welcome_message,
            logo_url=assistant.logo_url,
            primary_color=assistant.primary_color,
        )


class AskWidgetQuestion:
    def __init__(
        self,
        *,
        assistant_repository: AssistantRepository,
        embedding_provider: EmbeddingProvider,
        vector_repository: VectorRepository,
        llm_provider: LLMProvider,
        prompt_builder: GroundedPromptBuilder,
        top_k: int,
        similarity_threshold: float,
    ) -> None:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not isfinite(similarity_threshold) or not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self._assistant_repository = assistant_repository
        self._embedding_provider = embedding_provider
        self._vector_repository = vector_repository
        self._llm_provider = llm_provider
        self._prompt_builder = prompt_builder
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    async def execute(self, command: AskWidgetQuestionCommand) -> Answer:
        assistant = await self._assistant_repository.get_by_id(command.assistant_id)
        if assistant is None:
            raise AssistantNotFoundError("Assistant not found")

        cleaned_question = command.question.strip()
        if not cleaned_question:
            raise ValueError("question must not be blank")

        query_embedding = await self._embedding_provider.embed_query(cleaned_question)
        retrieved_chunks = await self._vector_repository.search_similar(
            command.assistant_id,
            query_embedding,
            limit=self._top_k,
            minimum_similarity=self._similarity_threshold,
        )
        sufficient_chunks = self._select_sufficient_chunks(retrieved_chunks)

        if not sufficient_chunks:
            if is_conversational_greeting(cleaned_question):
                greeting_instruction = build_greeting_prompt(
                    assistant_name=assistant.name,
                    assistant_instructions=assistant.assistant_instructions,
                )
                greeting_response = (
                    await self._llm_provider.generate(
                        system_instruction=greeting_instruction,
                        content=cleaned_question,
                    )
                ).strip()
                return Answer(text=greeting_response, sources=())

            return Answer(text=FALLBACK_ANSWER)

        grounded_prompt = self._prompt_builder.build(
            question=cleaned_question,
            chunks=sufficient_chunks,
            assistant_instructions=assistant.assistant_instructions,
        )
        answer_text = (
            await self._llm_provider.generate(
                system_instruction=grounded_prompt.system_instruction,
                content=grounded_prompt.content,
            )
        ).strip()
        if answer_text == FALLBACK_ANSWER:
            return Answer(text=FALLBACK_ANSWER)

        return Answer(
            text=answer_text,
            sources=self._collect_sources(sufficient_chunks),
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
