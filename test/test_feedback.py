"""Tests for the feedback database layer.

Requires a running PostgreSQL instance with pgvector.
Run: docker compose -f pgvector.yaml up -d
Then: alembic upgrade head
Then: python -m pytest test/test_feedback.py -v
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from database.connection import get_session
from database.repository import DocumentRepository


@pytest.fixture()
def session():
    """Provide a database session that rolls back after each test."""
    sess = get_session()
    yield sess
    sess.rollback()
    sess.close()


@pytest.fixture()
def repo():
    """Provide a DocumentRepository instance."""
    return DocumentRepository()


class TestFeedbackLog:
    """Tests for feedback log insertion and stats."""

    def test_insert_feedback_returns_uuid(self, session, repo):
        """Insert a thumbs_up feedback record and verify the returned ID."""
        feedback_id = repo.insert_feedback_log(session, {
            "query_log_id": str(uuid.uuid4()),
            "feedback_type": "thumbs_up",
            "rating": 5,
            "query_text": "What are polymers?",
        })
        assert feedback_id
        assert len(feedback_id) == 36  # UUID string length

    def test_feedback_stats_counts(self, session, repo):
        """Insert multiple feedback records and verify aggregated stats."""
        repo.insert_feedback_log(session, {
            "feedback_type": "thumbs_up",
            "rating": 5,
        })
        repo.insert_feedback_log(session, {
            "feedback_type": "thumbs_up",
            "rating": 4,
        })
        repo.insert_feedback_log(session, {
            "feedback_type": "thumbs_down",
            "rating": 1,
        })
        repo.insert_feedback_log(session, {
            "feedback_type": "correction",
            "correction": "The answer should mention X.",
            "rating": 2,
        })
        session.flush()

        stats = repo.get_feedback_stats(session)
        assert stats["total"] == 4
        assert stats["thumbs_up"] == 2
        assert stats["thumbs_down"] == 1
        assert stats["correction"] == 1
        assert stats["avg_rating"] == pytest.approx(3.0)

    def test_feedback_stats_since_future_returns_zero(self, session, repo):
        """Filtering with a future datetime returns zero counts."""
        repo.insert_feedback_log(session, {
            "feedback_type": "thumbs_up",
            "rating": 5,
        })
        session.flush()

        future = datetime.now(timezone.utc) + timedelta(days=1)
        stats = repo.get_feedback_stats(session, since=future)
        assert stats["total"] == 0
        assert stats["thumbs_up"] == 0
        assert stats["thumbs_down"] == 0
        assert stats["correction"] == 0
        assert stats["avg_rating"] is None
