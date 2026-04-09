"""Evaluation and feedback endpoints."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from database.connection import get_session
from database.repository import DocumentRepository
from evaluation.feedback import EvaluationSummary, FeedbackRequest, FeedbackResponse
from observability.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/feedback",
    status_code=201,
    response_model=FeedbackResponse,
)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit user feedback for a RAG query response."""
    session = get_session()
    try:
        repo = DocumentRepository()
        feedback_id = repo.insert_feedback_log(session, {
            "query_log_id": request.query_log_id,
            "feedback_type": request.feedback_type.value,
            "rating": request.rating,
            "correction": request.correction,
            "query_text": request.query_text,
        })
        session.commit()

        logger.info(
            "Feedback submitted",
            extra={
                "feedback_id": feedback_id,
                "feedback_type": request.feedback_type.value,
                "query_log_id": request.query_log_id,
            },
        )

        # Re-fetch the record to get the server-generated created_at
        from database.models import FeedbackLogModel
        from sqlalchemy import select

        stmt = select(FeedbackLogModel).where(FeedbackLogModel.id == feedback_id)
        record = session.execute(stmt).scalar_one()

        return FeedbackResponse(id=feedback_id, created_at=record.created_at)
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.warning("Failed to submit feedback: %s", e)
        raise HTTPException(status_code=503, detail="Failed to submit feedback")
    finally:
        session.close()


@router.get(
    "/evaluation/summary",
    response_model=EvaluationSummary,
)
def evaluation_summary(since: str | None = None) -> EvaluationSummary:
    """Get aggregated feedback and evaluation statistics."""
    since_dt: datetime | None = None
    if since is not None:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid 'since' parameter. Use ISO 8601 format.",
            )

    session = get_session()
    try:
        repo = DocumentRepository()
        stats = repo.get_feedback_stats(session, since=since_dt)

        total = stats["total"]
        thumbs_up = stats["thumbs_up"]
        positive_rate = thumbs_up / total if total > 0 else 0.0

        return EvaluationSummary(
            total_feedback=total,
            positive_rate=round(positive_rate, 4),
            avg_rating=stats["avg_rating"],
            counts_by_type={
                "thumbs_up": stats["thumbs_up"],
                "thumbs_down": stats["thumbs_down"],
                "correction": stats["correction"],
            },
            since=since_dt,
        )
    finally:
        session.close()
