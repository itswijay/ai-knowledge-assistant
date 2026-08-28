import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from app.application.use_cases import IngestDocumentCommand
from app.core.config import get_settings
from app.dependencies import build_application_container
from app.domain.entities import SourceReference
from app.evaluation.evaluator import RAGEvaluator
from app.evaluation.models import EvaluationCase, ExpectedBehavior
from app.evaluation.sample_pdf import build_sample_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = PROJECT_ROOT / "evaluations"
DEFAULT_DATASET = EVALUATION_ROOT / "data" / "sample_product_guide.json"
DEFAULT_OUTPUT = EVALUATION_ROOT / "results" / "latest.json"
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DatasetCase(BaseModel):
    question: NonEmptyText
    expected_behavior: ExpectedBehavior
    expected_answer_terms: list[NonEmptyText] = Field(default_factory=list)
    expected_page: int | None = Field(default=None, ge=1)


class Dataset(BaseModel):
    document: NonEmptyText
    pages: list[list[NonEmptyText]] = Field(min_length=1)
    cases: list[DatasetCase] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    document: str
    pages: tuple[tuple[str, ...], ...]
    cases: tuple[EvaluationCase, ...]


def load_evaluation_suite(path: Path) -> EvaluationSuite:
    dataset = Dataset.model_validate_json(path.read_text(encoding="utf-8"))
    cases = tuple(
        EvaluationCase(
            question=case.question,
            expected_behavior=case.expected_behavior,
            expected_answer_terms=tuple(case.expected_answer_terms),
            expected_source=(
                SourceReference(document=dataset.document, page=case.expected_page)
                if case.expected_page is not None
                else None
            ),
        )
        for case in dataset.cases
    )
    return EvaluationSuite(
        document=dataset.document,
        pages=tuple(tuple(page) for page in dataset.pages),
        cases=cases,
    )


async def run_evaluation(
    suite: EvaluationSuite,
    *,
    user_id: UUID,
    assistant_id: UUID,
    output_path: Path,
    ingest_sample: bool,
) -> bool:
    container = build_application_container(get_settings())
    try:
        if ingest_sample:
            await container.ingest_document.execute(
                IngestDocumentCommand(
                    user_id=user_id,
                    assistant_id=assistant_id,
                    filename=suite.document,
                    content=build_sample_pdf(suite.pages),
                )
            )
        report = await RAGEvaluator(
            container.ask_question,
            user_id=user_id,
            assistant_id=assistant_id,
        ).evaluate(suite.cases)
    finally:
        await container.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report.summary
    print(f"Report: {output_path}")
    print(f"Answerable accuracy: {summary.answerable_accuracy:.1%}")
    print(f"Refusal accuracy: {summary.refusal_accuracy:.1%}")
    print(f"Source accuracy: {summary.source_accuracy:.1%}")
    return (
        summary.answerable_target_met
        and summary.refusal_target_met
        and summary.source_target_met
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation")
    parser.add_argument(
        "--user-id",
        type=UUID,
        required=True,
        help="Authenticated owner or admin user UUID",
    )
    parser.add_argument(
        "--assistant-id",
        type=UUID,
        required=True,
        help="Assistant UUID used for ingestion",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Evaluate an already-ingested copy of the sample document",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_evaluation_suite(args.dataset)
    targets_met = asyncio.run(
        run_evaluation(
            suite,
            user_id=args.user_id,
            assistant_id=args.assistant_id,
            output_path=args.output,
            ingest_sample=not args.skip_ingestion,
        )
    )
    return 0 if targets_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
