from pathlib import Path

from app.evaluation import ExpectedBehavior
from app.evaluation.run import load_evaluation_suite
from app.evaluation.sample_pdf import build_sample_pdf
from app.infrastructure.documents import PyPdfDocumentParser

DATASET = (
    Path(__file__).resolve().parents[3]
    / "evaluations"
    / "data"
    / "sample_product_guide.json"
)


def test_sample_suite_has_required_case_mix_and_sources() -> None:
    suite = load_evaluation_suite(DATASET)

    answerable = [
        case
        for case in suite.cases
        if case.expected_behavior is ExpectedBehavior.ANSWER
    ]
    unanswerable = [
        case
        for case in suite.cases
        if case.expected_behavior is ExpectedBehavior.REFUSE
    ]
    assert len(answerable) >= 10
    assert len(unanswerable) >= 10
    assert all(case.expected_source is not None for case in answerable)


def test_generated_sample_pdf_preserves_pages_and_text() -> None:
    suite = load_evaluation_suite(DATASET)

    pages = PyPdfDocumentParser().parse(build_sample_pdf(suite.pages))

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert "two-year limited warranty" in pages[0].content
    assert "800 lumens" in pages[1].content
    assert "2.4 GHz Wi-Fi" in pages[2].content
