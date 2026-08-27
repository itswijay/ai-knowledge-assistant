from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceReference:
    document: str
    page: int

    def __post_init__(self) -> None:
        if not self.document.strip():
            raise ValueError("document must not be blank")
        if self.page < 1:
            raise ValueError("page must be at least 1")


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[SourceReference, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be blank")
