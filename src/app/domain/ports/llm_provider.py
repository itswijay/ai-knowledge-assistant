from typing import Protocol


class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...
