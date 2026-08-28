from typing import Protocol


class LLMProvider(Protocol):
    async def generate(self, *, system_instruction: str, content: str) -> str: ...
