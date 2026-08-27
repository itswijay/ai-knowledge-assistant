from typing import Protocol


class DocumentValidator(Protocol):
    def validate(self, filename: str, content: bytes) -> str: ...
