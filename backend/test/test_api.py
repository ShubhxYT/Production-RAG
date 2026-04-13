"""Tests for the FastAPI API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from generation.models import TokenUsage
from pipeline.models import LatencyBreakdown, RAGResponse, SourceCitation

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    @patch("api.routes.health.get_session")
    def test_health_ok(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["database"] == "connected"

    @patch("api.routes.health.get_session")
    def test_health_db_disconnected(self, mock_get_session):
        mock_get_session.side_effect = Exception("Connection refused")

        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["database"] == "disconnected"


# ---------------------------------------------------------------------------
# Documents endpoint
# ---------------------------------------------------------------------------


class TestDocumentsEndpoint:
    """Tests for GET /documents."""

    @patch("api.routes.documents.get_session")
    def test_documents_returns_list(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = []
        mock_get_session.return_value = mock_session

        response = client.get("/documents")
        assert response.status_code == 200
        body = response.json()
        assert "documents" in body
        assert "total" in body
        assert body["total"] == 0


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


def _mock_rag_response() -> RAGResponse:
    return RAGResponse(
        answer="Test answer about polymers.",
        sources=[
            SourceCitation(
                document_title="Polymer Lecture",
                source_path="data/polymer.md",
                chunk_summary="About polymers.",
                page_numbers=[1],
                similarity_score=0.85,
            )
        ],
        latency=LatencyBreakdown(
            retrieval_ms=10.0,
            context_ms=1.0,
            generation_ms=200.0,
            total_ms=211.0,
        ),
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        prompt_version="qa_v1",
    )


class TestQueryEndpoint:
    """Tests for POST /query."""

    @patch("api.routes.query._get_pipeline")
    def test_query_success(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.query = AsyncMock(return_value=_mock_rag_response())
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post(
            "/query",
            json={"question": "What are polymers?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Test answer about polymers."
        assert len(body["sources"]) == 1
        assert body["prompt_version"] == "qa_v1"
        assert body["latency"]["total_ms"] > 0

    @patch("api.routes.query._get_pipeline")
    def test_query_with_top_k(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.query = AsyncMock(return_value=_mock_rag_response())
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post(
            "/query",
            json={"question": "Test?", "top_k": 3},
        )
        assert response.status_code == 200
        mock_pipeline.query.assert_called_once_with(
            question="Test?", top_k=3, prompt_variant=None
        )

    @patch("api.routes.query._get_pipeline")
    def test_query_with_prompt_variant(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.query = AsyncMock(return_value=_mock_rag_response())
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post(
            "/query",
            json={"question": "Summarize.", "prompt_variant": "summarize"},
        )
        assert response.status_code == 200
        mock_pipeline.query.assert_called_once_with(
            question="Summarize.", top_k=5, prompt_variant="summarize"
        )

    def test_query_empty_question_returns_422(self):
        response = client.post(
            "/query",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_query_missing_question_returns_422(self):
        response = client.post(
            "/query",
            json={},
        )
        assert response.status_code == 422

    def test_query_too_long_question_returns_422(self):
        response = client.post(
            "/query",
            json={"question": "x" * 1001},
        )
        assert response.status_code == 422

    @patch("api.routes.query._get_pipeline")
    def test_query_pipeline_failure_returns_503(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("LLM unavailable")
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post(
            "/query",
            json={"question": "Test?"},
        )
        assert response.status_code == 503
