from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.domain.entities import Organization, OrganizationMember, OrganizationRole
from app.domain.errors import ResourceConflictError, TenantRepositoryError
from app.infrastructure.database.models import (
    OrganizationMemberModel,
    OrganizationModel,
)
from app.infrastructure.database.session import AsyncSessionFactory


class PostgresOrganizationRepository:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def create_with_owner(
        self,
        organization: Organization,
        owner: OrganizationMember,
    ) -> None:
        if owner.organization_id != organization.id:
            raise ValueError("Owner membership must belong to the organization")
        if owner.role is not OrganizationRole.OWNER:
            raise ValueError("Initial organization membership must be owner")

        organization_model = self._to_model(organization)
        organization_model.members.append(
            PostgresOrganizationMemberRepository._to_model(owner)
        )
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add(organization_model)
        except IntegrityError as error:
            raise ResourceConflictError(
                "Organization conflicts with existing data"
            ) from error
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to persist organization") from error

    async def list_for_user(self, user_id: UUID) -> Sequence[Organization]:
        statement = (
            select(OrganizationModel)
            .join(
                OrganizationMemberModel,
                OrganizationMemberModel.organization_id == OrganizationModel.id,
            )
            .where(OrganizationMemberModel.user_id == user_id)
            .order_by(OrganizationModel.created_at.asc(), OrganizationModel.id.asc())
        )
        try:
            async with self._session_factory() as session:
                result = await session.execute(statement)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to list organizations") from error
        return tuple(self._to_domain(model) for model in result.scalars().all())

    async def get_by_id(self, organization_id: UUID) -> Organization | None:
        try:
            async with self._session_factory() as session:
                model = await session.get(OrganizationModel, organization_id)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to retrieve organization") from error
        return self._to_domain(model) if model is not None else None

    @staticmethod
    def _to_model(organization: Organization) -> OrganizationModel:
        return OrganizationModel(
            id=organization.id,
            name=organization.name,
            created_at=organization.created_at,
            updated_at=organization.updated_at,
        )

    @staticmethod
    def _to_domain(model: OrganizationModel) -> Organization:
        return Organization(
            id=model.id,
            name=model.name,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


class PostgresOrganizationMemberRepository:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def get_membership(
        self,
        organization_id: UUID,
        user_id: UUID,
    ) -> OrganizationMember | None:
        try:
            async with self._session_factory() as session:
                model = await session.get(
                    OrganizationMemberModel,
                    (organization_id, user_id),
                )
        except SQLAlchemyError as error:
            raise TenantRepositoryError(
                "Unable to retrieve organization membership"
            ) from error
        return self._to_domain(model) if model is not None else None

    @staticmethod
    def _to_model(membership: OrganizationMember) -> OrganizationMemberModel:
        return OrganizationMemberModel(
            organization_id=membership.organization_id,
            user_id=membership.user_id,
            role=membership.role.value,
            created_at=membership.created_at,
        )

    @staticmethod
    def _to_domain(model: OrganizationMemberModel) -> OrganizationMember:
        return OrganizationMember(
            organization_id=model.organization_id,
            user_id=model.user_id,
            role=OrganizationRole(model.role),
            created_at=model.created_at,
        )
