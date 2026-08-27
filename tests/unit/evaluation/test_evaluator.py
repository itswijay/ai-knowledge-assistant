from dataclasses import dataclass, field
from uuid import UUID

import pytest

from app.application.constants import FALLBACK_ANSWER
from app.application.use_cases import QuestionAnswerTrace
from app.domain.entities import Answer, RetrievedChunk, SourceReference
from app.evaluation import EvaluationCase, ExpectedBehavior, RAGEvaluator


@dataclass
class FakeQuestionAnswerer:
    traces: dict[str, QuestionAnswerTrace]
    calls: list[str] = field(default_factory=list)

    async def execute_with_trace(self, question: str) -> QuestionAnswerTrace:
        self.calls.append(question)
        return self.traces[question]


def retrieved_chunk(
    *,
    document: str,
    page: int,
    similarity: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=page),
        document_id=UUID(int=100),
        original_filename=document,
        page_number=page,
        chunk_index=page - 1,
        content="Evaluation context",
        similarity_score=similarity,
    )


@pytest.mark.asyncio
async def test_evaluator_records_answers_sources_scores_and_targets() -> None:
    source = SourceReference(document="guide.pdf", page=2)
    answerable = EvaluationCase(
        question="How bright is it?",
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_answer_terms=("800", "lumens"),
        expected_source=source,
    )
    unanswerable = EvaluationCase(
        question="How much does it weigh?",
        expected_behavior=ExpectedBehavior.REFUSE,
    )
    answerer = FakeQuestionAnswerer(
        traces={
            answerable.question: QuestionAnswerTrace(
                answer=Answer(
                    text="The lamp produces up to 800 lumens.",
                    sources=(source,),
                ),
                retrieved_chunks=(
                    retrieved_chunk(document="guide.pdf", page=2, similarity=0.91),
                ),
            ),
            unanswerable.question: QuestionAnswerTrace(
                answer=Answer(text=FALLBACK_ANSWER),
                retrieved_chunks=(
                    retrieved_chunk(document="guide.pdf", page=1, similarity=0.32),
                ),
            ),
        }
    )

    report = await RAGEvaluator(answerer).evaluate([answerable, unanswerable])

    assert answerer.calls == [answerable.question, unanswerable.question]
    assert report.records[0].correctness is True
    assert report.records[0].expected_source == source
    assert report.records[0].retrieved_source == source
    assert report.records[0].similarity_score == 0.91
    assert report.records[0].source_correct is True
    assert report.records[1].correctness is True
    assert report.records[1].similarity_score == 0.32
    assert report.records[1].source_correct is None
    assert report.summary.answerable_accuracy == 1.0
    assert report.summary.refusal_accuracy == 1.0
    assert report.summary.source_accuracy == 1.0
    assert report.summary.answerable_target_met is True
    assert report.summary.refusal_target_met is True
    assert report.summary.source_target_met is True


@pytest.mark.asyncio
async def test_evaluator_marks_incorrect_answer_and_source() -> None:
    expected_source = SourceReference(document="guide.pdf", page=1)
    case = EvaluationCase(
        question="How long is the warranty?",
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_answer_terms=("two-year",),
        expected_source=expected_source,
    )
    refusal_case = EvaluationCase(
        question="What is the price?",
        expected_behavior=ExpectedBehavior.REFUSE,
    )
    answerer = FakeQuestionAnswerer(
        traces={
            case.question: QuestionAnswerTrace(
                answer=Answer(
                    text="The warranty is one year.",
                    sources=(SourceReference(document="other.pdf", page=4),),
                ),
                retrieved_chunks=(),
            ),
            refusal_case.question: QuestionAnswerTrace(
                answer=Answer(
                    text="The price is $50.",
                    sources=(expected_source,),
                ),
                retrieved_chunks=(),
            ),
        }
    )

    report = await RAGEvaluator(answerer).evaluate([case, refusal_case])

    assert report.records[0].correctness is False
    assert report.records[0].source_correct is False
    assert report.records[1].correctness is False
    assert report.summary.answerable_target_met is False
    assert report.summary.refusal_target_met is False
    assert report.summary.source_target_met is False


@pytest.mark.asyncio
async def test_evaluator_requires_both_case_types() -> None:
    case = EvaluationCase(
        question="How bright is it?",
        expected_behavior=ExpectedBehavior.ANSWER,
        expected_answer_terms=("800",),
        expected_source=SourceReference(document="guide.pdf", page=2),
    )
    answerer = FakeQuestionAnswerer(
        traces={
            case.question: QuestionAnswerTrace(
                answer=Answer(text="800 lumens", sources=(case.expected_source,)),
                retrieved_chunks=(),
            )
        }
    )

    with pytest.raises(ValueError, match="answerable and unanswerable"):
        await RAGEvaluator(answerer).evaluate([case])
