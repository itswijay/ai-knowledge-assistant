from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.application.use_cases import (
    AskWidgetQuestion,
    AskWidgetQuestionCommand,
    GetWidgetConfig,
)
from app.dependencies import get_ask_widget_question, get_widget_config
from app.presentation.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatSourceResponse,
)
from app.presentation.api.schemas.widget import WidgetConfigResponse

router = APIRouter(prefix="/api/v1/widget", tags=["widget"])


@router.get("/{assistant_id}/config", response_model=WidgetConfigResponse)
async def get_config(
    assistant_id: UUID,
    use_case: Annotated[GetWidgetConfig, Depends(get_widget_config)],
) -> WidgetConfigResponse:
    config = await use_case.execute(assistant_id)
    return WidgetConfigResponse(
        id=config.id,
        name=config.name,
        welcome_message=config.welcome_message,
        logo_url=config.logo_url,
        primary_color=config.primary_color,
    )


@router.post("/{assistant_id}/chat", response_model=ChatResponse)
async def chat(
    assistant_id: UUID,
    request: ChatRequest,
    use_case: Annotated[AskWidgetQuestion, Depends(get_ask_widget_question)],
) -> ChatResponse:
    answer = await use_case.execute(
        AskWidgetQuestionCommand(
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
