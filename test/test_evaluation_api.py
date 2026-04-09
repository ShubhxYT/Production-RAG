"""Tests for the evaluation and feedback API endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestFeedbackEndpoint:
    """Tests for POST /feedback."""

    @patch("api.routes.evaluation.get_session")
    def test_submit_feedback_success(self, mock_get_session):
        """Submit valid thumbs_up feedback and verify 201 response."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        feedback_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        now = datetime.now(timezone.utc)

        # Mock the repository insert
        with patch("api.routes.evaluation.DocumentRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.insert_feedback_log.return_value = feedback_id

            # Mock the re-fetch to get created_at
            mock_record = MagicMock()
            mock_record.created_at = now
            mock_result = MagicMock()
            mock_result.scalar_one.return_value = mock_record
            mock_session.execute.return_value = mock_result

            response = client.post(
                "/feedback",
                json={
                    "feedback_type": "thumbs_up",
                    "rating": 5,
                    "query_text": "What are polymers?",
                },
            )

        assert response.status_code == 201
        body = response.json()
        assert body["id"] == feedback_id
        assert "created_at" in body

    def test_submit_feedback_invalid_rating_returns_422(self):
        """Rating outside 1-5 range should return 422."""
        response = client.post(
            "/feedback",
            json={
                "feedback_type": "thumbs_up",
                "rating": 6,
            },
        )
        assert response.status_code == 422

    def test_submit_feedback_invalid_type_returns_422(self):
        """Invalid feedback_type should return 422."""
        response = client.post(
            "/feedback",
            json={
                "feedback_type": "invalid_type",
            },
        )
        assert response.status_code == 422


class TestEvaluationSummaryEndpoint:
    """Tests for GET /evaluation/summary."""

    @patch("api.routes.evaluation.get_session")
    def test_summary_success(self, mock_get_session):
        """Get evaluation summary with mocked stats."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        with patch("api.routes.evaluation.DocumentRepository") as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_feedback_stats.return_value = {
                "total": 10,
                "thumbs_up": 7,
                "thumbs_down": 2,
                "correction": 1,
                "avg_rating": 4.2,
            }

            response = client.get("/evaluation/summary")

        assert response.status_code == 200
        body = response.json()
        assert body["total_feedback"] == 10
        assert body["positive_rate"] == 0.7
        assert body["avg_rating"] == 4.2
        assert body["counts_by_type"]["thumbs_up"] == 7
        assert body["since"] is None

    def test_summary_invalid_since_returns_400(self):
        """Invalid since parameter should return 400."""
        response = client.get("/evaluation/summary?since=not-a-date")
        assert response.status_code == 400
