from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.application.use_cases import AskQuestion, AskQuestionCommand
from app.dependencies import get_ask_question, get_authenticated_user
from app.domain.entities import AuthenticatedUser
from app.presentation.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
)

router = APIRouter(prefix="/api/v1/assistants", tags=["chat"])


@router.post("/{assistant_id}/chat", response_model=ChatResponse)
async def chat(
    assistant_id: UUID,
    request: ChatRequest,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[AskQuestion, Depends(get_ask_question)],
) -> ChatResponse:
    answer = await use_case.execute(
        AskQuestionCommand(
            user_id=user.id,
            assistant_id=assistant_id,
            question=request.message,
        )
    )
    return ChatResponse(
        answer=answer.text,
        sources=[
            ChatSourceResponse(document=source.document, page=source.page)
            for source in answer.sources
        ],
    )
