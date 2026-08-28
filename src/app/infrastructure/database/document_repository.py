from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.domain.entities import Document
from app.domain.errors import TenantRepositoryError
from app.infrastructure.database.models import DocumentModel
from app.infrastructure.database.session import AsyncSessionFactory


class PostgresDocumentRepository:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def list_by_assistant(self, assistant_id: UUID) -> Sequence[Document]:
        statement = (
            select(DocumentModel)
            .where(DocumentModel.assistant_id == assistant_id)
            .order_by(DocumentModel.created_at.asc(), DocumentModel.id.asc())
        )
        try:
            async with self._session_factory() as session:
                result = await session.execute(statement)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to list documents") from error
        return tuple(self._to_domain(model) for model in result.scalars().all())

    async def get_by_id(self, document_id: UUID) -> Document | None:
        try:
            async with self._session_factory() as session:
                model = await session.get(DocumentModel, document_id)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to retrieve document") from error
        return self._to_domain(model) if model is not None else None

    async def delete(self, document_id: UUID) -> bool:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    model = await session.get(DocumentModel, document_id)
                    if model is None:
                        return False
                    await session.delete(model)
        except SQLAlchemyError as error:
            raise TenantRepositoryError("Unable to delete document") from error
        return True

    @staticmethod
    def _to_domain(model: DocumentModel) -> Document:
        return Document(
            id=model.id,
            assistant_id=model.assistant_id,
            original_filename=model.original_filename,
            created_at=model.created_at,
        )
