from collections.abc import Sequence
from typing import Protocol

from app.application.constants import FALLBACK_ANSWER
from app.application.use_cases import QuestionAnswerTrace
from app.domain.entities import SourceReference
from app.evaluation.models import (
    EvaluationCase,
    EvaluationRecord,
    EvaluationReport,
    EvaluationSummary,
    ExpectedBehavior,
)

ANSWERABLE_ACCURACY_TARGET = 0.9
REFUSAL_ACCURACY_TARGET = 1.0
SOURCE_ACCURACY_TARGET = 0.9


class TracedQuestionAnswerer(Protocol):
    async def execute_with_trace(self, question: str) -> QuestionAnswerTrace: ...


class RAGEvaluator:
    def __init__(self, question_answerer: TracedQuestionAnswerer) -> None:
        self._question_answerer = question_answerer

    async def evaluate(self, cases: Sequence[EvaluationCase]) -> EvaluationReport:
        if not cases:
            raise ValueError("at least one evaluation case is required")

        records = tuple([await self._evaluate_case(case) for case in cases])
        return EvaluationReport(records=records, summary=self._summarize(records))

    async def _evaluate_case(self, case: EvaluationCase) -> EvaluationRecord:
        trace = await self._question_answerer.execute_with_trace(case.question)
        retrieved_source, similarity_score = self._top_retrieval(trace)

        if case.expected_behavior is ExpectedBehavior.REFUSE:
            correctness = (
                trace.answer.text == FALLBACK_ANSWER and not trace.answer.sources
            )
            source_correct = None
        else:
            normalized_answer = trace.answer.text.casefold()
            correctness = trace.answer.text != FALLBACK_ANSWER and all(
                term.casefold() in normalized_answer
                for term in case.expected_answer_terms
            )
            source_correct = case.expected_source in trace.answer.sources

        return EvaluationRecord(
            question=case.question,
            expected_behavior=case.expected_behavior,
            actual_answer=trace.answer.text,
            correctness=correctness,
            expected_source=case.expected_source,
            retrieved_source=retrieved_source,
            similarity_score=similarity_score,
            source_correct=source_correct,
        )

    @staticmethod
    def _top_retrieval(
        trace: QuestionAnswerTrace,
    ) -> tuple[SourceReference | None, float | None]:
        if not trace.retrieved_chunks:
            return None, None
        top_chunk = max(
            trace.retrieved_chunks,
            key=lambda chunk: chunk.similarity_score,
        )
        return (
            SourceReference(
                document=top_chunk.original_filename,
                page=top_chunk.page_number,
            ),
            top_chunk.similarity_score,
        )

    @staticmethod
    def _summarize(records: Sequence[EvaluationRecord]) -> EvaluationSummary:
        answerable = [
            record
            for record in records
            if record.expected_behavior is ExpectedBehavior.ANSWER
        ]
        unanswerable = [
            record
            for record in records
            if record.expected_behavior is ExpectedBehavior.REFUSE
        ]
        if not answerable or not unanswerable:
            raise ValueError("evaluation requires answerable and unanswerable cases")

        answerable_accuracy = sum(record.correctness for record in answerable) / len(
            answerable
        )
        refusal_accuracy = sum(record.correctness for record in unanswerable) / len(
            unanswerable
        )
        source_accuracy = sum(
            record.source_correct is True for record in answerable
        ) / len(answerable)
        return EvaluationSummary(
            answerable_count=len(answerable),
            unanswerable_count=len(unanswerable),
            answerable_accuracy=answerable_accuracy,
            refusal_accuracy=refusal_accuracy,
            source_accuracy=source_accuracy,
            answerable_target_met=answerable_accuracy >= ANSWERABLE_ACCURACY_TARGET,
            refusal_target_met=refusal_accuracy >= REFUSAL_ACCURACY_TARGET,
            source_target_met=source_accuracy >= SOURCE_ACCURACY_TARGET,
        )
