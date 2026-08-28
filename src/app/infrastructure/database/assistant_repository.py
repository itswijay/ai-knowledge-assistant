from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.domain.entities import Assistant
from app.domain.errors import (
    AssistantNotFoundError,
    ResourceConflictError,
    TenantRepositoryError,
)
from app.infrastructure.database.models import AssistantModel
from app.infrastructure.database.session import AsyncSessionFactory


class PostgresAssistantRepository:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def create(self, assistant: Assistant) -> None:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add(self._to_model(assistant))
        except IntegrityError as error:
            raise ResourceConflictError(
                "Assistant conflicts with existing data"
            ) from error
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to persist assistant") from error

    async def list_by_organization(
        self,
        organization_id: UUID,
    ) -> Sequence[Assistant]:
        statement = (
            select(AssistantModel)
            .where(AssistantModel.organization_id == organization_id)
            .order_by(AssistantModel.created_at.asc(), AssistantModel.id.asc())
        )
        try:
            async with self._session_factory() as session:
                result = await session.execute(statement)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to list assistants") from error
        return tuple(self._to_domain(model) for model in result.scalars().all())

    async def get_by_id(self, assistant_id: UUID) -> Assistant | None:
        try:
            async with self._session_factory() as session:
                model = await session.get(AssistantModel, assistant_id)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to retrieve assistant") from error
        return self._to_domain(model) if model is not None else None

    async def update(self, assistant: Assistant) -> None:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    model = await session.get(AssistantModel, assistant.id)
                    if model is None:
                        raise AssistantNotFoundError("Assistant not found")
                    if model.organization_id != assistant.organization_id:
                        raise ValueError(
                            "Assistant organization ownership cannot be changed"
                        )
                    self._apply(model, assistant)
        except IntegrityError as error:
            raise ResourceConflictError(
                "Assistant conflicts with existing data"
            ) from error
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to update assistant") from error

    async def delete(self, assistant_id: UUID) -> bool:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    model = await session.get(AssistantModel, assistant_id)
                    if model is None:
                        return False
                    await session.delete(model)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to delete assistant") from error
        return True

    @staticmethod
    def _to_model(assistant: Assistant) -> AssistantModel:
        return AssistantModel(
            id=assistant.id,
            organization_id=assistant.organization_id,
            name=assistant.name,
            description=assistant.description,
            welcome_message=assistant.welcome_message,
            system_prompt=assistant.system_prompt,
            logo_url=assistant.logo_url,
            primary_color=assistant.primary_color,
            created_at=assistant.created_at,
            updated_at=assistant.updated_at,
        )

    @staticmethod
    def _to_domain(model: AssistantModel) -> Assistant:
        return Assistant(
            id=model.id,
            organization_id=model.organization_id,
            name=model.name,
            description=model.description,
            welcome_message=model.welcome_message,
            system_prompt=model.system_prompt,
            logo_url=model.logo_url,
            primary_color=model.primary_color,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _apply(model: AssistantModel, assistant: Assistant) -> None:
        model.name = assistant.name
        model.description = assistant.description
        model.welcome_message = assistant.welcome_message
        model.system_prompt = assistant.system_prompt
        model.logo_url = assistant.logo_url
        model.primary_color = assistant.primary_color
        model.updated_at = assistant.updated_at
