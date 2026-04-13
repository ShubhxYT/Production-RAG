"""Health check endpoint."""

from fastapi import APIRouter
from observability.logging import get_logger
from sqlalchemy import text

from api.models import ErrorResponse, HealthResponse
from database.connection import get_session

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
)
def health_check() -> HealthResponse:
    """Check API and database health."""
    try:
        session = get_session()
        try:
            session.execute(text("SELECT 1"))
            db_status = "connected"
        finally:
            session.close()
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        db_status = "disconnected"

    return HealthResponse(status="ok", database=db_status)
