from collections.abc import Sequence
from typing import Protocol

from app.domain.entities import DocumentPage


class DocumentParser(Protocol):
    def parse(self, content: bytes) -> Sequence[DocumentPage]: ...
