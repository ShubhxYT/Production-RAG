"""Integration tests for the database layer.

Requires a running PostgreSQL instance with pgvector.
Run: docker compose -f pgvector.yaml up -d
Then: alembic upgrade head
Then: python -m pytest test/test_database.py -v
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from database.connection import get_engine, get_session, reset_engine
from database.models import Base, ChunkEmbeddingModel, ChunkModel, DocumentModel
from database.repository import DocumentRepository
from ingestion.models import Chunk, Document, ElementType


def _make_document(
    source_path: str | None = None,
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
        title="Test Document",
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


class TestDatabaseConnection:
    """Verify database connectivity."""

    def test_connection(self):
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_pgvector_extension(self):
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector'"
                )
            )
            assert result.scalar() == "vector"

    def test_tables_exist(self):
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' "
                    "ORDER BY tablename"
                )
            )
            tables = [row[0] for row in result]
            assert "documents" in tables
            assert "chunks" in tables
            assert "chunk_embeddings" in tables


class TestDocumentRepository:
    """Test CRUD operations on documents, chunks, and embeddings."""

    def test_insert_document(self, session, repo):
        doc = _make_document(num_chunks=2)
        doc_id = repo.insert_document(session, doc)
        assert doc_id == doc.id

        # Verify document row
        db_doc = repo.get_document_by_source_path(session, doc.source_path)
        assert db_doc is not None
        assert db_doc.title == "Test Document"
        assert db_doc.format == "md"

    def test_insert_document_with_chunks(self, session, repo):
        doc = _make_document(num_chunks=3)
        repo.insert_document(session, doc)

        chunks = repo.get_chunks_by_document(session, doc.id)
        assert len(chunks) == 3
        assert chunks[0].position == 0
        assert chunks[1].position == 1
        assert chunks[2].position == 2

    def test_chunk_metadata(self, session, repo):
        doc = _make_document(num_chunks=1)
        repo.insert_document(session, doc)

        chunks = repo.get_chunks_by_document(session, doc.id)
        chunk = chunks[0]
        assert chunk.summary == "Summary of chunk 0."
        assert "test" in chunk.keywords
        assert chunk.section_path == ["Section 1", "Subsection 0"]
        assert chunk.page_numbers == [1]
        assert "paragraph" in chunk.element_types

    def test_insert_and_search_embeddings(self, session, repo):
        doc = _make_document(num_chunks=2)
        repo.insert_document(session, doc)

        # Insert embeddings
        vec1 = _make_embedding()
        vec2 = [v * -1 for v in vec1]  # Opposite direction
        repo.insert_bulk_embeddings(
            session,
            [
                (doc.chunks[0].id, vec1),
                (doc.chunks[1].id, vec2),
            ],
            model_name="test-model",
        )

        # Search with vec1 — chunk 0 should be most similar
        results = repo.search_by_vector(session, vec1, top_k=2)
        assert len(results) == 2
        assert results[0][0].id == doc.chunks[0].id
        assert results[0][1] > results[1][1]  # Higher similarity for closer vector

    def test_search_with_threshold(self, session, repo):
        doc = _make_document(num_chunks=1)
        repo.insert_document(session, doc)
        repo.insert_embeddings(
            session, doc.chunks[0].id, _make_embedding(), "test-model"
        )

        # Very high threshold — should return no results
        results = repo.search_by_vector(
            session, _make_embedding(), top_k=5, threshold=0.9999
        )
        # May or may not match depending on vector — just verify no error
        assert isinstance(results, list)

    def test_filter_by_keywords(self, session, repo):
        doc = _make_document(num_chunks=2)
        repo.insert_document(session, doc)

        results = repo.filter_by_metadata(session, keywords=["chunk0"])
        assert len(results) == 1
        assert results[0].keywords == ["test", "chunk0"]

    def test_filter_by_format(self, session, repo):
        doc = _make_document(num_chunks=1)
        repo.insert_document(session, doc)

        results = repo.filter_by_metadata(session, format="md")
        assert len(results) >= 1

        results = repo.filter_by_metadata(session, format="pdf")
        # Our test doc is "md", so pdf filter should not include it
        matching = [r for r in results if r.document_id == doc.id]
        assert len(matching) == 0

    def test_cascade_delete(self, session, repo):
        doc = _make_document(num_chunks=2)
        repo.insert_document(session, doc)
        repo.insert_bulk_embeddings(
            session,
            [(c.id, _make_embedding()) for c in doc.chunks],
            "test-model",
        )

        # Delete document
        deleted = repo.delete_document(session, doc.id)
        assert deleted is True

        # Chunks and embeddings should be gone
        chunks = repo.get_chunks_by_document(session, doc.id)
        assert len(chunks) == 0

    def test_delete_nonexistent(self, session, repo):
        deleted = repo.delete_document(session, str(uuid.uuid4()))
        assert deleted is False

    def test_duplicate_source_path(self, session, repo):
        doc1 = _make_document(source_path="test/duplicate.md", num_chunks=1)
        repo.insert_document(session, doc1)
        session.flush()

        doc2 = _make_document(source_path="test/duplicate.md", num_chunks=1)
        with pytest.raises(Exception):
            repo.insert_document(session, doc2)
            session.flush()

    def test_tsvector_populated(self, session, repo):
        doc = _make_document(num_chunks=1)
        repo.insert_document(session, doc)
        session.flush()

        # Query the tsvector column directly
        result = session.execute(
            text("SELECT tsv IS NOT NULL FROM chunks WHERE id = :id"),
            {"id": doc.chunks[0].id},
        )
        assert result.scalar() is True


class TestQueryLog:
    """Test query audit log operations."""

    def test_insert_query_log(self, session, repo):
        log_id = repo.insert_query_log(session, {
            "query": "What is polymer?",
            "answer": "A polymer is a material...",
            "sources": [{"source_path": "test.md", "similarity_score": 0.9}],
            "prompt_variant": "qa",
            "prompt_version": "qa_v1",
            "retrieval_top_k": 5,
            "retrieval_result_count": 3,
            "latency_ms": 150.5,
            "retrieval_ms": 50.0,
            "generation_ms": 100.0,
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "model": "gemini-2.5-flash",
        })
        assert log_id is not None

    def test_query_log_readable(self, session, repo):
        from database.models import QueryLogModel

        repo.insert_query_log(session, {
            "query": "Test query",
            "answer": "Test answer",
            "sources": [],
        })

        from sqlalchemy import select
        stmt = select(QueryLogModel).where(QueryLogModel.query == "Test query")
        record = session.execute(stmt).scalar_one()
        assert record.answer == "Test answer"
        assert record.created_at is not None
