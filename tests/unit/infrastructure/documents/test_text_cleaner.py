from app.infrastructure.documents.text_cleaner import clean_text


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("  Warranty\n\tcoverage   lasts two years.  ") == (
        "Warranty coverage lasts two years."
    )


def test_clean_text_handles_empty_content() -> None:
    assert clean_text(" \n\t ") == ""
