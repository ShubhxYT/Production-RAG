"""Pydantic models for feedback request/response and evaluation summary."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Allowed feedback types."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"


class FeedbackRequest(BaseModel):
    """POST /feedback request body."""

    query_log_id: str | None = None
    feedback_type: FeedbackType
    rating: int | None = Field(default=None, ge=1, le=5)
    correction: str | None = Field(default=None, max_length=2000)
    query_text: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    """POST /feedback response body."""

    id: str
    created_at: datetime


class EvaluationSummary(BaseModel):
    """GET /evaluation/summary response body."""

    total_feedback: int
    positive_rate: float
    avg_rating: float | None
    counts_by_type: dict[str, int]
    since: datetime | None
