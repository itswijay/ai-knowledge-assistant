import re

WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize extracted whitespace without altering textual content."""

    return WHITESPACE_PATTERN.sub(" ", text).strip()
