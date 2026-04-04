"""SQLAlchemy 2.0 declarative models for documents, chunks, and embeddings."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class DocumentModel(Base):
    """Ingested document with metadata."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    source_path = Column(String, nullable=False, unique=True)
    format = Column(String, nullable=False)
    raw_content_hash = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    doc_version = Column(String, nullable=True)
    doc_date = Column(String, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)

    chunks = relationship(
        "ChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_documents_source_path", "source_path"),
    )


class ChunkModel(Base):
    """A chunk of text with enrichment metadata."""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        UUID(as_uuid=False),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    text = Column(Text, nullable=False)
    summary = Column(Text, nullable=False, default="")
    keywords = Column(ARRAY(Text), nullable=False, default=list)
    hypothetical_questions = Column(ARRAY(Text), nullable=False, default=list)
    section_path = Column(ARRAY(Text), nullable=False, default=list)
    page_numbers = Column(ARRAY(Integer), nullable=False, default=list)
    element_types = Column(ARRAY(Text), nullable=False, default=list)
    token_count = Column(Integer, nullable=False, default=0)
    position = Column(Integer, nullable=False, default=0)
    overlap_before = Column(Text, nullable=False, default="")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Full-text search vector -- generated column for Step 15 hybrid search
    tsv = Column(
        TSVECTOR,
        nullable=True,
    )

    document = relationship("DocumentModel", back_populates="chunks")
    embeddings = relationship(
        "ChunkEmbeddingModel",
        back_populates="chunk",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_created_at", "created_at"),
        Index("ix_chunks_keywords", "keywords", postgresql_using="gin"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )


class ChunkEmbeddingModel(Base):
    """Vector embedding for a chunk."""

    __tablename__ = "chunk_embeddings"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_id = Column(
        UUID(as_uuid=False),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding = Column(Vector(768), nullable=False)
    embedding_model = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    chunk = relationship("ChunkModel", back_populates="embeddings")

    __table_args__ = (
        Index(
            "ix_chunk_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class QueryLogModel(Base):
    """Audit log for RAG pipeline queries."""

    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String, nullable=True, index=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(JSONB, nullable=False, default=list)
    prompt_variant = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True)
    retrieval_top_k = Column(Integer, nullable=True)
    retrieval_result_count = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    retrieval_ms = Column(Float, nullable=True)
    generation_ms = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    model = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
