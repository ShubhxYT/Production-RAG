"""Tests for the evaluation runner.

Requires a running PostgreSQL instance with pgvector.
Run: docker compose -f pgvector.yaml up -d
Then: alembic upgrade head
Then: python -m pytest test/test_evaluation_runner.py -v
"""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from database.connection import get_session
from database.repository import DocumentRepository
from embeddings.models import EmbeddingConfig, EmbeddingResult
from embeddings.service import EmbeddingService
from evaluation.models import EvaluationReport, GroundTruthDataset
from evaluation.retrieval_runner import EvaluationRunner
from ingestion.models import Chunk, Document, ElementType
from retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers (reused pattern from test_retrieval.py)
# ---------------------------------------------------------------------------


def _make_document(
    source_path: str | None = None,
    title: str = "Eval Test Document",
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
                text=f"Test chunk {i} about evaluation metrics. " * 10,
                token_count=50,
                document_id=doc_id,
                section_path=["Section 1", f"Subsection {i}"],
                page_numbers=[i + 1],
                element_types=[ElementType.PARAGRAPH],
                position=i,
                overlap_before="" if i == 0 else f"overlap from chunk {i - 1}",
                summary=f"Summary of evaluation chunk {i}.",
                keywords=["evaluation", f"chunk{i}"],
                hypothetical_questions=[f"What is evaluation chunk {i}?"],
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
    """Mock embedding provider for testing."""

    def __init__(self, fixed_vector: list[float] | None = None, dim: int = 768) -> None:
        self.dim = dim
        self.fixed_vector = fixed_vector

    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResult:
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
    return DocumentRepository()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestGroundTruthDataset:
    def test_load_from_json(self):
        """Verify GroundTruthDataset can be loaded from a JSON string."""
        data = {
            "version": "0.1.0",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [
                {
                    "query": "test query",
                    "relevant_chunk_ids": ["id1", "id2"],
                    "tags": ["factual"],
                    "notes": "test note",
                }
            ],
        }
        ds = GroundTruthDataset.model_validate(data)
        assert ds.version == "0.1.0"
        assert len(ds.annotations) == 1
        assert ds.annotations[0].query == "test query"
        assert len(ds.annotations[0].relevant_chunk_ids) == 2

    def test_roundtrip_json(self):
        data = {
            "version": "1.0.0",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [],
        }
        ds = GroundTruthDataset.model_validate(data)
        json_str = ds.model_dump_json()
        restored = GroundTruthDataset.model_validate_json(json_str)
        assert restored.version == "1.0.0"


class TestEvaluationReport:
    def test_report_roundtrip(self):
        report = EvaluationReport(
            dataset_version="0.1.0",
            retrieval_config={"top_k": 5},
            aggregate_metrics=[],
            per_query_results=[],
        )
        json_str = report.model_dump_json()
        restored = EvaluationReport.model_validate_json(json_str)
        assert restored.dataset_version == "0.1.0"


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


class TestEvaluationRunner:
    def _setup(self, session, repo):
        """Insert test data and return (service, chunk_ids)."""
        doc = _make_document()
        repo.insert_document(session, doc)

        vec = _make_embedding()
        repo.insert_bulk_embeddings(
            session,
            [(c.id, vec) for c in doc.chunks],
            model_name="BAAI/bge-base-en-v1.5",
        )

        provider = MockProvider(fixed_vector=vec)
        config = EmbeddingConfig()
        embed_service = EmbeddingService(provider=provider, config=config)
        service = RetrievalService(embedding_service=embed_service, config=config)

        chunk_ids = [c.id for c in doc.chunks]
        return service, chunk_ids

    def test_run_produces_report(self, session, repo):
        """Runner produces a valid EvaluationReport."""
        service, chunk_ids = self._setup(session, repo)

        # Create a temp ground-truth dataset referencing actual chunk IDs
        dataset = {
            "version": "test",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [
                {
                    "query": "evaluation metrics",
                    "relevant_chunk_ids": [chunk_ids[0]],
                    "tags": ["factual"],
                },
                {
                    "query": "no match query",
                    "relevant_chunk_ids": ["nonexistent-id"],
                    "tags": ["negative"],
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dataset, f)
            dataset_path = Path(f.name)

        runner = EvaluationRunner(
            retrieval_service=service,
            dataset_path=dataset_path,
            top_k=5,
            k_values=[1, 3, 5],
        )

        with patch("retrieval.service.get_session", return_value=session):
            report = runner.run()

        assert isinstance(report, EvaluationReport)
        assert report.dataset_version == "test"
        assert len(report.per_query_results) == 2
        assert len(report.aggregate_metrics) > 0

        # Metrics should include precision, recall, mrr, ndcg for each k
        metric_names = {m.metric_name for m in report.aggregate_metrics}
        assert "precision@5" in metric_names
        assert "recall@5" in metric_names
        assert "mrr" in metric_names
        assert "ndcg@5" in metric_names

        dataset_path.unlink()

    def test_save_and_load_report(self, session, repo):
        """Report can be saved to disk and loaded back."""
        service, chunk_ids = self._setup(session, repo)

        dataset = {
            "version": "save-test",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [
                {
                    "query": "simple query",
                    "relevant_chunk_ids": [chunk_ids[0]],
                    "tags": [],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dataset, f)
            dataset_path = Path(f.name)

        runner = EvaluationRunner(
            retrieval_service=service,
            dataset_path=dataset_path,
            top_k=5,
            k_values=[5],
        )

        with patch("retrieval.service.get_session", return_value=session):
            report = runner.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = EvaluationRunner.save_report(report, output_dir=Path(tmpdir))
            assert out_path.exists()
            assert out_path.suffix == ".json"

            # Verify it loads back
            loaded = EvaluationReport.model_validate_json(
                out_path.read_text(encoding="utf-8")
            )
            assert loaded.dataset_version == "save-test"

        dataset_path.unlink()

    def test_empty_dataset(self, session, repo):
        """Runner handles an empty dataset gracefully."""
        service, _ = self._setup(session, repo)

        dataset = {
            "version": "empty",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dataset, f)
            dataset_path = Path(f.name)

        runner = EvaluationRunner(
            retrieval_service=service,
            dataset_path=dataset_path,
            top_k=5,
        )

        with patch("retrieval.service.get_session", return_value=session):
            report = runner.run()

        assert len(report.per_query_results) == 0
        assert len(report.aggregate_metrics) == 0

        dataset_path.unlink()
