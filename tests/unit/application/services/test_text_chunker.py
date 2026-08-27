import pytest

from app.application.services import WordChunker
from app.domain.entities import DocumentPage


def test_chunker_splits_words_with_overlap() -> None:
    page = DocumentPage(
        page_number=3,
        content="one two three four five six seven eight nine ten eleven twelve",
    )
    chunker = WordChunker(chunk_size_words=5, overlap_words=2)

    chunks = chunker.chunk_pages([page])

    assert [chunk.content for chunk in chunks] == [
        "one two three four five",
        "four five six seven eight",
        "seven eight nine ten eleven",
        "ten eleven twelve",
    ]
    assert [chunk.page_number for chunk in chunks] == [3, 3, 3, 3]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]


def test_chunker_does_not_mix_pages() -> None:
    pages = (
        DocumentPage(page_number=1, content="one two three four"),
        DocumentPage(page_number=2, content="five six seven four"),
    )
    chunker = WordChunker(chunk_size_words=3, overlap_words=1)

    chunks = chunker.chunk_pages(pages)

    assert [(chunk.page_number, chunk.content) for chunk in chunks] == [
        (1, "one two three"),
        (1, "three four"),
        (2, "five six seven"),
        (2, "seven four"),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunker_rejects_invalid_configuration(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        WordChunker(chunk_size_words=chunk_size, overlap_words=overlap)
