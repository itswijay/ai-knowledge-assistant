from typing import Annotated

from fastapi import APIRouter, Depends

from app.application.use_cases import AskQuestion
from app.dependencies import get_ask_question
from app.presentation.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    use_case: Annotated[AskQuestion, Depends(get_ask_question)],
) -> ChatResponse:
    answer = await use_case.execute(request.message)
    return ChatResponse(
        answer=answer.text,
        sources=[
            ChatSourceResponse(document=source.document, page=source.page)
            for source in answer.sources
        ],
    )
