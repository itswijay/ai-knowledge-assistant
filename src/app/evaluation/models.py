from dataclasses import dataclass
from enum import StrEnum

from app.domain.entities import SourceReference


class ExpectedBehavior(StrEnum):
    ANSWER = "answer"
    REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    question: str
    expected_behavior: ExpectedBehavior
    expected_answer_terms: tuple[str, ...] = ()
    expected_source: SourceReference | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if self.expected_behavior is ExpectedBehavior.ANSWER:
            if not self.expected_answer_terms:
                raise ValueError("answerable cases require expected answer terms")
            if self.expected_source is None:
                raise ValueError("answerable cases require an expected source")
        elif self.expected_answer_terms or self.expected_source is not None:
            raise ValueError("refusal cases must not define answer terms or a source")
        if any(not term.strip() for term in self.expected_answer_terms):
            raise ValueError("expected answer terms must not be blank")


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    question: str
    expected_behavior: ExpectedBehavior
    actual_answer: str
    correctness: bool
    expected_source: SourceReference | None
    retrieved_source: SourceReference | None
    similarity_score: float | None
    source_correct: bool | None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    answerable_count: int
    unanswerable_count: int
    answerable_accuracy: float
    refusal_accuracy: float
    source_accuracy: float
    answerable_target_met: bool
    refusal_target_met: bool
    source_target_met: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    records: tuple[EvaluationRecord, ...]
    summary: EvaluationSummary
