from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status

from app.application.use_cases import (
    CreateAssistant,
    CreateAssistantCommand,
    DeleteAssistant,
    GetAssistant,
    ListAssistants,
    UpdateAssistant,
    UpdateAssistantCommand,
)
from app.dependencies import (
    get_authenticated_user,
    get_create_assistant,
    get_delete_assistant,
    get_get_assistant,
    get_list_assistants,
    get_update_assistant,
)
from app.domain.entities import Assistant, AuthenticatedUser
from app.presentation.api.schemas.assistants import (
    AssistantResponse,
    CreateAssistantRequest,
    UpdateAssistantRequest,
)

router = APIRouter(prefix="/api/v1", tags=["assistants"])


@router.post(
    "/organizations/{organization_id}/assistants",
    response_model=AssistantResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_assistant(
    organization_id: UUID,
    request: CreateAssistantRequest,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[CreateAssistant, Depends(get_create_assistant)],
) -> AssistantResponse:
    assistant = await use_case.execute(
        CreateAssistantCommand(
            user_id=user.id,
            organization_id=organization_id,
            name=request.name,
            description=request.description,
            welcome_message=request.welcome_message,
            assistant_instructions=request.assistant_instructions,
            logo_url=request.logo_url,
            primary_color=request.primary_color,
        )
    )
    return _to_response(assistant)


@router.get(
    "/organizations/{organization_id}/assistants",
    response_model=list[AssistantResponse],
)
async def list_assistants(
    organization_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[ListAssistants, Depends(get_list_assistants)],
) -> list[AssistantResponse]:
    assistants = await use_case.execute(
        user_id=user.id,
        organization_id=organization_id,
    )
    return [_to_response(assistant) for assistant in assistants]


@router.get("/assistants/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(
    assistant_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[GetAssistant, Depends(get_get_assistant)],
) -> AssistantResponse:
    assistant = await use_case.execute(user_id=user.id, assistant_id=assistant_id)
    return _to_response(assistant)


@router.patch("/assistants/{assistant_id}", response_model=AssistantResponse)
async def update_assistant(
    assistant_id: UUID,
    request: UpdateAssistantRequest,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[UpdateAssistant, Depends(get_update_assistant)],
) -> AssistantResponse:
    updates = request.model_dump(exclude_unset=True)
    assistant = await use_case.execute(
        UpdateAssistantCommand(
            user_id=user.id,
            assistant_id=assistant_id,
            **updates,
        )
    )
    return _to_response(assistant)


@router.delete(
    "/assistants/{assistant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_assistant(
    assistant_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(get_authenticated_user)],
    use_case: Annotated[DeleteAssistant, Depends(get_delete_assistant)],
) -> Response:
    await use_case.execute(user_id=user.id, assistant_id=assistant_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _to_response(assistant: Assistant) -> AssistantResponse:
    return AssistantResponse(
        id=assistant.id,
        organization_id=assistant.organization_id,
        name=assistant.name,
        description=assistant.description,
        welcome_message=assistant.welcome_message,
        assistant_instructions=assistant.assistant_instructions,
        logo_url=assistant.logo_url,
        primary_color=assistant.primary_color,
        created_at=assistant.created_at,
        updated_at=assistant.updated_at,
    )
