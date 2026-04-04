"""Documents listing endpoint."""

import logging

from fastapi import APIRouter
from sqlalchemy import func, select

from api.models import DocumentInfo, DocumentsResponse, ErrorResponse
from database.connection import get_session
from database.models import ChunkModel, DocumentModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentsResponse,
    responses={500: {"model": ErrorResponse}},
)
def list_documents() -> DocumentsResponse:
    """List all ingested documents with chunk counts."""
    session = get_session()
    try:
        stmt = (
            select(
                DocumentModel.id,
                DocumentModel.title,
                DocumentModel.source_path,
                DocumentModel.created_at,
                func.count(ChunkModel.id).label("chunk_count"),
            )
            .outerjoin(ChunkModel, ChunkModel.document_id == DocumentModel.id)
            .group_by(DocumentModel.id)
            .order_by(DocumentModel.created_at.desc())
        )
        rows = session.execute(stmt).all()

        documents = [
            DocumentInfo(
                id=str(row.id),
                title=row.title,
                source_path=row.source_path,
                chunk_count=row.chunk_count,
                created_at=row.created_at.isoformat(),
            )
            for row in rows
        ]

        return DocumentsResponse(documents=documents, total=len(documents))
    finally:
        session.close()
