from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

ChatMessage = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]


class ChatRequest(BaseModel):
    message: ChatMessage


class ChatSourceResponse(BaseModel):
    document: str = Field(min_length=1)
    page: int = Field(ge=1)


class ChatResponse(BaseModel):
    answer: str = Field(min_length=1)
    sources: list[ChatSourceResponse]
