"""Tests for the retrieval service.

Requires a running PostgreSQL instance with pgvector.
Run: docker compose -f pgvector.yaml up -d
Then: alembic upgrade head
Then: python -m pytest test/test_retrieval.py -v
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from database.connection import get_session
from database.repository import DocumentRepository
from embeddings.models import EmbeddingConfig, EmbeddingResult
from embeddings.service import EmbeddingService
from ingestion.models import Chunk, Document, ElementType
from retrieval.models import RetrievalResponse, RetrievalResult
from retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(
    source_path: str | None = None,
    title: str = "Test Document",
    num_chunks: int = 3,
) -> Document:
    """Create a test Document with chunks."""
    doc_id = str(uuid.uuid4())
    source_path = source_path or f"test/{doc_id}.md"
    chunks = []
    for i in range(num_chunks):
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=f"Test chunk {i} content for document {doc_id}. " * 10,
                token_count=50,
                document_id=doc_id,
                section_path=["Section 1", f"Subsection {i}"],
                page_numbers=[i + 1],
                element_types=[ElementType.PARAGRAPH],
                position=i,
                overlap_before="" if i == 0 else f"overlap from chunk {i - 1}",
                summary=f"Summary of chunk {i}.",
                keywords=["test", f"chunk{i}"],
                hypothetical_questions=[f"What is chunk {i}?"],
            )
        )
    return Document(
        id=doc_id,
        source_path=source_path,
        title=title,
        format="md",
        raw_content="Full raw content here.",
        chunks=chunks,
        created_at=datetime.now(timezone.utc),
    )


def _make_embedding(dim: int = 768) -> list[float]:
    """Create a random-ish embedding vector."""
    import random

    random.seed(42)
    return [random.uniform(-1, 1) for _ in range(dim)]


class MockProvider:
    """Mock embedding provider for testing (no GPU required)."""

    def __init__(self, fixed_vector: list[float] | None = None, dim: int = 768) -> None:
        self.dim = dim
        self.fixed_vector = fixed_vector
        self.call_count = 0

    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResult:
        self.call_count += 1
        if self.fixed_vector:
            vectors = [self.fixed_vector for _ in texts]
        else:
            vectors = [[0.1] * self.dim for _ in texts]
        return EmbeddingResult(
            vectors=vectors,
            model=config.model_name,
            dimensions=self.dim,
            token_usage=0,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestRetrievalModels:
    """Test RetrievalResult and RetrievalResponse models."""

    def test_retrieval_result_roundtrip(self):
        result = RetrievalResult(
            chunk_id="abc-123",
            text="Hello world",
            summary="A greeting.",
            keywords=["hello", "world"],
            section_path=["Chapter 1"],
            page_numbers=[1],
            document_id="doc-456",
            document_title="My Doc",
            source_path="test/my_doc.md",
            similarity_score=0.95,
            token_count=10,
        )
        data = result.model_dump()
        restored = RetrievalResult.model_validate(data)
        assert restored.chunk_id == "abc-123"
        assert restored.similarity_score == 0.95
        assert restored.keywords == ["hello", "world"]

    def test_retrieval_result_defaults(self):
        result = RetrievalResult(
            chunk_id="x",
            text="t",
            summary="s",
            document_id="d",
            source_path="p",
            similarity_score=0.5,
        )
        assert result.keywords == []
        assert result.section_path == []
        assert result.page_numbers == []
        assert result.document_title is None
        assert result.token_count == 0

    def test_retrieval_response_roundtrip(self):
        response = RetrievalResponse(
            query="test query",
            top_k=5,
            threshold=0.7,
            result_count=1,
            latency_ms=42.5,
            results=[
                RetrievalResult(
                    chunk_id="c1",
                    text="text",
                    summary="summary",
                    document_id="d1",
                    source_path="path",
                    similarity_score=0.9,
                )
            ],
        )
        data = response.model_dump()
        restored = RetrievalResponse.model_validate(data)
        assert restored.query == "test query"
        assert restored.result_count == 1
        assert len(restored.results) == 1

    def test_retrieval_response_empty(self):
        response = RetrievalResponse(
            query="nothing",
            top_k=5,
            result_count=0,
            latency_ms=10.0,
        )
        assert response.results == []
        assert response.threshold is None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRetrievalService:
    """Integration tests for the retrieval service against a real database."""

    def _setup_service(
        self, session, repo, fixed_vector: list[float] | None = None
    ) -> tuple[RetrievalService, Document]:
        """Insert a test document with embeddings and create a service.

        Returns the service and the inserted document.
        """
        doc = _make_document(title="Polymer Lecture Notes")
        repo.insert_document(session, doc)

        # Insert embeddings for each chunk
        vec = _make_embedding()
        repo.insert_bulk_embeddings(
            session,
            [(c.id, vec) for c in doc.chunks],
            model_name="BAAI/bge-base-en-v1.5",
        )

        # Build service with mock provider returning the same vector
        provider = MockProvider(fixed_vector=fixed_vector or vec)
        config = EmbeddingConfig()
        embed_service = EmbeddingService(provider=provider, config=config)

        # Patch get_session to return our test session
        service = RetrievalService(embedding_service=embed_service, config=config)
        return service, doc

    def test_retrieve_returns_results(self, session, repo):
        service, doc = self._setup_service(session, repo)

        with patch("retrieval.service.get_session", return_value=session):
            response = service.retrieve("test query", top_k=5)

        assert response.result_count > 0
        assert response.query == "test query"
        assert response.top_k == 5
        assert response.latency_ms > 0

    def test_retrieve_result_fields(self, session, repo):
        service, doc = self._setup_service(session, repo)

        with patch("retrieval.service.get_session", return_value=session):
            response = service.retrieve("test query", top_k=5)

        result = response.results[0]
        assert result.chunk_id in [c.id for c in doc.chunks]
        assert result.document_id == doc.id
        assert result.document_title == "Polymer Lecture Notes"
        assert result.source_path == doc.source_path
        assert result.similarity_score > 0
        assert result.text != ""
        assert result.summary != ""
        assert isinstance(result.keywords, list)
        assert isinstance(result.section_path, list)

    def test_retrieve_top_k_limits_results(self, session, repo):
        service, doc = self._setup_service(session, repo)

        with patch("retrieval.service.get_session", return_value=session):
            response = service.retrieve("test query", top_k=2)

        assert response.result_count <= 2

    def test_retrieve_with_threshold(self, session, repo):
        service, doc = self._setup_service(session, repo)

        # Very high threshold - may filter out results
        with patch("retrieval.service.get_session", return_value=session):
            response = service.retrieve(
                "test query", top_k=5, threshold=0.9999
            )

        assert response.threshold == 0.9999
        # All returned results must meet the threshold
        for result in response.results:
            assert result.similarity_score >= 0.9999

    def test_retrieve_no_matching_documents(self, session, repo):
        """Test with an orthogonal vector that won't match well."""
        vec = _make_embedding()
        # Create a vector that's very different
        opposite_vec = [-v for v in vec]

        service, doc = self._setup_service(
            session, repo, fixed_vector=opposite_vec
        )

        with patch("retrieval.service.get_session", return_value=session):
            response = service.retrieve(
                "completely unrelated query", top_k=5, threshold=0.99
            )

        # With a high threshold and opposite vector, should get few/no results
        for result in response.results:
            assert result.similarity_score >= 0.99

    def test_retrieve_results_sorted_by_similarity(self, session, repo):
        service, doc = self._setup_service(session, repo)

        with patch("retrieval.service.get_session", return_value=session):
            response = service.retrieve("test query", top_k=5)

        if len(response.results) > 1:
            scores = [r.similarity_score for r in response.results]
            assert scores == sorted(scores, reverse=True)

    def test_retrieve_response_metadata(self, session, repo):
        service, doc = self._setup_service(session, repo)

        with patch("retrieval.service.get_session", return_value=session):
            response = service.retrieve("hello", top_k=3, threshold=0.5)

        assert response.query == "hello"
        assert response.top_k == 3
        assert response.threshold == 0.5
        assert response.latency_ms >= 0
        assert response.result_count == len(response.results)