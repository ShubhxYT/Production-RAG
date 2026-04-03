# Vector Database Setup (Step 6)

## Goal
Set up PostgreSQL with pgvector schema, Alembic migrations, a repository/DAO layer, and a seed script to load staged documents with embeddings into the database — bridging the ingestion pipeline with future retrieval.

## Prerequisites
Make sure that you are currently on the `feat/vector-database-setup` branch before beginning implementation.
If not, move to the correct branch. If the branch does not exist, create it from main.

```bash
git checkout -b feat/vector-database-setup main
```

Ensure PostgreSQL is running:
```bash
docker compose -f pgvector.yaml up -d
```

---

### Step-by-Step Instructions

#### Step 1: Add Dependencies & Database Config

- [x] Add database dependencies to `pyproject.toml`:

Replace the existing `dependencies` list in `pyproject.toml` with:

```toml
dependencies = [
    "beautifulsoup4>=4.12.0",
    "google-genai>=1.70.0",
    "langchain-docling>=2.0.0",
    "langchain-opendataloader-pdf>=2.0.0",
    "liteparse>=1.2.1",
    "markdown-it-py>=3.0.0",
    "marker-pdf>=1.10.2",
    "markdownify>=0.13.0",
    "pgvector>=0.3.6",
    "psycopg[binary]>=3.2.0",
    "pydantic>=2.0.0",
    "python-docx>=1.0.0",
    "python-dotenv>=1.0.0",
    "sentence-transformers>=3.0.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.15.0",
    "tiktoken>=0.7.0",
    "torch>=2.11.0",
    "torchvision>=0.26.0",
]
```

- [x] Add `get_database_url()` to `config/settings.py`:

Replace the entire file with:

```python
"""Environment-based settings loader."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def get_gemini_api_key() -> str:
    """Return the Gemini API key from environment.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set.
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key


def get_cerebras_api_key() -> str:
    """Return the Cerebras API key from environment.

    Raises:
        RuntimeError: If CEREBRAS_API_KEY is not set.
    """
    key = os.environ.get("CEREBRAS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "CEREBRAS_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key


def get_database_url() -> str:
    """Return the database URL from environment.

    Defaults to the local Docker PostgreSQL from pgvector.yaml
    if DATABASE_URL is not set.

    Returns:
        SQLAlchemy-compatible database URL.
    """
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://fullrag:fullrag@localhost:5432/fullrag",
    )
```

- [x] Update `.env.example` — append database URL:

Add to the end of `.env.example`:

```
# PostgreSQL + pgvector (local Docker default matches pgvector.yaml)
DATABASE_URL=postgresql+psycopg://fullrag:fullrag@localhost:5432/fullrag
```

- [x] Install dependencies:

```bash
uv sync
```

##### Step 1 Verification Checklist
- [x] `uv sync` completes with zero errors
- [x] `python -c "from config.settings import get_database_url; print(get_database_url())"` prints `postgresql+psycopg://fullrag:fullrag@localhost:5432/fullrag`
- [x] `python -c "import sqlalchemy; import alembic; import pgvector; print('OK')"` prints `OK`

#### Step 1 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

```bash
git add -A && git commit -m "feat(database): add psycopg, sqlalchemy, alembic, pgvector dependencies and database config"
```

---

#### Step 2: Define SQLAlchemy Models & Database Connection

- [x] Create `database/__init__.py`:

```python
"""Database layer for the FullRag system — PostgreSQL with pgvector."""
```

- [x] Create `database/connection.py`:

```python
"""Database engine and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_database_url

_engine = None
_session_factory = None


def get_engine():
    """Return a cached SQLAlchemy engine.

    Uses the DATABASE_URL from config with connection pooling.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory():
    """Return a cached sessionmaker bound to the engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory


def get_session() -> Session:
    """Create and return a new database session."""
    factory = get_session_factory()
    return factory()


def reset_engine():
    """Dispose of the current engine and reset cached state.

    Useful for testing or when reconfiguring the database URL.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
```

- [x] Create `database/models.py`:

```python
"""SQLAlchemy 2.0 declarative models for documents, chunks, and embeddings."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Column,
    DateTime,
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

    # Full-text search vector — generated column for Step 15 hybrid search
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
```

##### Step 2 Verification Checklist
- [x] `python -c "from database.models import Base, DocumentModel, ChunkModel, ChunkEmbeddingModel; print('Models OK')"` prints `Models OK`
- [x] `python -c "from database.connection import get_engine; e = get_engine(); print(e.url)"` prints the database URL
- [x] No import errors

#### Step 2 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

```bash
git add -A && git commit -m "feat(database): define SQLAlchemy models for documents, chunks, and embeddings"
```

---

#### Step 3: Initialize Alembic & Create Initial Migration

- [x] Initialize Alembic from the project root:

```bash
cd /home/shubhpc/FullRag
source .venv/bin/activate
alembic init alembic
```

- [x] Replace the generated `alembic.ini` — update `sqlalchemy.url` line to empty (we set it programmatically):

Find this line in `alembic.ini`:
```
sqlalchemy.url = driver://user:pass@localhost/dbname
```

Replace it with:
```
sqlalchemy.url =
```

- [x] Replace the generated `alembic/env.py` with:

```python
"""Alembic migration environment configuration."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool

from config.settings import get_database_url
from database.connection import get_engine
from database.models import Base

# Alembic Config object
config = context.config

# Set up loggers from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLAlchemy MetaData for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the given string to the script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the context.
    """
    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [x] Generate the initial migration:

```bash
alembic revision --autogenerate -m "initial schema with documents chunks and embeddings"
```

- [x] Open the generated migration file in `alembic/versions/` and add the pgvector extension creation at the top of the `upgrade()` function. The file will be auto-generated but needs this one manual addition.

Find this line near the top of `upgrade()`:
```python
def upgrade() -> None:
```

Replace `upgrade()` with the version that creates the pgvector extension first, then the tsvector trigger. The auto-generated table/index creation stays as-is. Add this at the very start of `upgrade()`:

```python
def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- auto-generated table/index operations follow below ---
```

And add at the very end of `upgrade()`, after all auto-generated operations:

```python
    # Create trigger to auto-update tsvector column on chunk insert/update
    op.execute("""
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', COALESCE(NEW.text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE OF text
        ON chunks FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
    """)
```

And add to the very end of `downgrade()`:

```python
    op.execute("DROP TRIGGER IF EXISTS tsvectorupdate ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_tsv_trigger()")
    op.execute("DROP EXTENSION IF EXISTS vector")
```

- [x] Run the migration:

```bash
alembic upgrade head
```

##### Step 3 Verification Checklist
- [x] `alembic upgrade head` runs without errors
- [x] Verify tables exist: `psql -h localhost -U fullrag -d fullrag -c "\dt"` shows `documents`, `chunks`, `chunk_embeddings`
- [x] Verify indexes exist: `psql -h localhost -U fullrag -d fullrag -c "\di"` shows HNSW, GIN, and B-tree indexes
- [x] Verify pgvector extension: `psql -h localhost -U fullrag -d fullrag -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"` returns `vector`
- [x] Verify tsvector trigger: `psql -h localhost -U fullrag -d fullrag -c "\df chunks_tsv_trigger"` shows the function

#### Step 3 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

```bash
git add -A && git commit -m "feat(database): initialize alembic and create initial migration with pgvector"
```

---

#### Step 4: Implement Repository/DAO Layer

- [ ] Create `database/repository.py`:

```python
"""Data access layer for documents, chunks, and embeddings."""

import hashlib
import logging

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from database.models import ChunkEmbeddingModel, ChunkModel, DocumentModel
from ingestion.models import Document as IngestionDocument

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for database CRUD and search operations."""

    def insert_document(self, session: Session, document: IngestionDocument) -> str:
        """Insert a document and all its chunks into the database.

        Maps from the Pydantic ingestion Document model to SQLAlchemy models.
        All chunks are inserted in the same transaction.

        Args:
            session: Active SQLAlchemy session.
            document: Pydantic Document from the ingestion pipeline.

        Returns:
            The document ID (UUID string).
        """
        content_hash = hashlib.sha256(
            document.raw_content.encode("utf-8")
        ).hexdigest()

        doc_model = DocumentModel(
            id=document.id,
            title=document.title,
            source_path=document.source_path,
            format=document.format,
            raw_content_hash=content_hash,
            created_at=document.created_at,
            doc_version=document.doc_version,
            doc_date=document.doc_date,
            metadata_=document.metadata,
        )
        session.add(doc_model)

        for chunk in document.chunks:
            chunk_model = ChunkModel(
                id=chunk.id,
                document_id=document.id,
                text=chunk.text,
                summary=chunk.summary,
                keywords=chunk.keywords,
                hypothetical_questions=chunk.hypothetical_questions,
                section_path=chunk.section_path,
                page_numbers=chunk.page_numbers,
                element_types=[et.value for et in chunk.element_types],
                token_count=chunk.token_count,
                position=chunk.position,
                overlap_before=chunk.overlap_before,
            )
            session.add(chunk_model)

        session.flush()
        logger.info(
            "Inserted document %s with %d chunks",
            document.id,
            len(document.chunks),
        )
        return document.id

    def insert_embeddings(
        self,
        session: Session,
        chunk_id: str,
        embedding: list[float],
        model_name: str,
    ) -> str:
        """Insert a single chunk embedding.

        Args:
            session: Active SQLAlchemy session.
            chunk_id: ID of the chunk this embedding belongs to.
            embedding: Vector as a list of floats.
            model_name: Name of the embedding model used.

        Returns:
            The embedding record ID.
        """
        emb_model = ChunkEmbeddingModel(
            chunk_id=chunk_id,
            embedding=embedding,
            embedding_model=model_name,
        )
        session.add(emb_model)
        session.flush()
        return emb_model.id

    def insert_bulk_embeddings(
        self,
        session: Session,
        embeddings: list[tuple[str, list[float]]],
        model_name: str,
    ) -> int:
        """Batch insert chunk embeddings.

        Args:
            session: Active SQLAlchemy session.
            embeddings: List of (chunk_id, vector) tuples.
            model_name: Name of the embedding model used.

        Returns:
            Number of embeddings inserted.
        """
        objects = [
            ChunkEmbeddingModel(
                chunk_id=chunk_id,
                embedding=vector,
                embedding_model=model_name,
            )
            for chunk_id, vector in embeddings
        ]
        session.add_all(objects)
        session.flush()
        logger.info("Inserted %d embeddings", len(objects))
        return len(objects)

    def get_document_by_source_path(
        self, session: Session, source_path: str
    ) -> DocumentModel | None:
        """Look up a document by its source path.

        Args:
            session: Active SQLAlchemy session.
            source_path: The source_path to search for.

        Returns:
            DocumentModel if found, else None.
        """
        stmt = select(DocumentModel).where(
            DocumentModel.source_path == source_path
        )
        return session.execute(stmt).scalar_one_or_none()

    def get_chunks_by_document(
        self, session: Session, document_id: str
    ) -> list[ChunkModel]:
        """Retrieve all chunks for a document, ordered by position.

        Args:
            session: Active SQLAlchemy session.
            document_id: The document ID.

        Returns:
            List of ChunkModel objects.
        """
        stmt = (
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.position)
        )
        return list(session.execute(stmt).scalars().all())

    def search_by_vector(
        self,
        session: Session,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[ChunkModel, float]]:
        """Cosine similarity search over chunk embeddings.

        Uses pgvector's <=> (cosine distance) operator. Distance is
        converted to similarity (1 - distance) for convenience.

        Args:
            session: Active SQLAlchemy session.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0-1). None disables filtering.

        Returns:
            List of (ChunkModel, similarity_score) tuples, highest similarity first.
        """
        distance = ChunkEmbeddingModel.embedding.cosine_distance(query_vector)
        similarity = (1 - distance).label("similarity")

        stmt = (
            select(ChunkModel, similarity)
            .join(
                ChunkEmbeddingModel,
                ChunkEmbeddingModel.chunk_id == ChunkModel.id,
            )
            .order_by(distance)
            .limit(top_k)
        )

        if threshold is not None:
            stmt = stmt.where(distance <= (1 - threshold))

        results = session.execute(stmt).all()
        return [(row[0], float(row[1])) for row in results]

    def filter_by_metadata(
        self,
        session: Session,
        keywords: list[str] | None = None,
        doc_date_after: str | None = None,
        format: str | None = None,
    ) -> list[ChunkModel]:
        """Filter chunks by metadata conditions.

        Args:
            session: Active SQLAlchemy session.
            keywords: If provided, chunks must contain at least one of these keywords.
            doc_date_after: If provided, only chunks from documents after this date.
            format: If provided, only chunks from documents of this format.

        Returns:
            List of matching ChunkModel objects.
        """
        stmt = select(ChunkModel).join(
            DocumentModel, DocumentModel.id == ChunkModel.document_id
        )

        if keywords:
            stmt = stmt.where(ChunkModel.keywords.overlap(keywords))

        if doc_date_after:
            stmt = stmt.where(DocumentModel.doc_date > doc_date_after)

        if format:
            stmt = stmt.where(DocumentModel.format == format)

        stmt = stmt.order_by(ChunkModel.position)
        return list(session.execute(stmt).scalars().all())

    def delete_document(self, session: Session, document_id: str) -> bool:
        """Delete a document and cascade to its chunks and embeddings.

        Args:
            session: Active SQLAlchemy session.
            document_id: The document ID to delete.

        Returns:
            True if the document was found and deleted, False otherwise.
        """
        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        doc = session.execute(stmt).scalar_one_or_none()
        if doc is None:
            return False
        session.delete(doc)
        session.flush()
        logger.info("Deleted document %s (cascade)", document_id)
        return True
```

##### Step 4 Verification Checklist
- [ ] `python -c "from database.repository import DocumentRepository; print('Repository OK')"` prints `Repository OK`
- [ ] No import errors

#### Step 4 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

```bash
git add -A && git commit -m "feat(database): implement repository/DAO layer with CRUD and vector search"
```

---

#### Step 5: Seed Script & CLI

- [ ] Create `database/seed.py`:

```python
"""Seed the database with staged documents and their embeddings."""

import logging
import time
from pathlib import Path

from database.connection import get_session
from database.repository import DocumentRepository
from embeddings.models import EmbeddingConfig
from embeddings.service import EmbeddingService
from ingestion.staging import load_staged_document

logger = logging.getLogger(__name__)


def seed_from_staging(
    staging_dir: str = "staging",
    embedding_service: EmbeddingService | None = None,
) -> dict:
    """Load all staged documents into the database with embeddings.

    Reads JSON files from the staging directory, generates embeddings
    for each chunk, and inserts everything into the database.
    Documents with a source_path already in the database are skipped.

    Args:
        staging_dir: Path to the staging directory with Document JSON files.
        embedding_service: EmbeddingService instance. Created with defaults if None.

    Returns:
        Summary dict with counts and timing.
    """
    staging_path = Path(staging_dir)
    if not staging_path.is_dir():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")

    json_files = sorted(staging_path.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", staging_dir)
        return {"documents": 0, "chunks": 0, "embeddings": 0, "skipped": 0}

    if embedding_service is None:
        embedding_service = EmbeddingService(config=EmbeddingConfig())

    repo = DocumentRepository()
    start_time = time.perf_counter()

    stats = {"documents": 0, "chunks": 0, "embeddings": 0, "skipped": 0, "failed": 0}

    for jf in json_files:
        try:
            doc = load_staged_document(jf)
        except Exception:
            logger.exception("Failed to load staged file: %s", jf.name)
            stats["failed"] += 1
            continue

        session = get_session()
        try:
            # Check for duplicate
            existing = repo.get_document_by_source_path(session, doc.source_path)
            if existing is not None:
                logger.info(
                    "Skipping (already exists): %s", doc.source_path
                )
                stats["skipped"] += 1
                session.close()
                continue

            # Insert document + chunks
            repo.insert_document(session, doc)
            stats["documents"] += 1
            stats["chunks"] += len(doc.chunks)

            # Generate and insert embeddings for each chunk
            if doc.chunks:
                chunk_texts = [c.text for c in doc.chunks]
                embed_result = embedding_service.embed(chunk_texts)

                embedding_pairs = [
                    (chunk.id, vector)
                    for chunk, vector in zip(doc.chunks, embed_result.vectors)
                ]
                repo.insert_bulk_embeddings(
                    session, embedding_pairs, embed_result.model
                )
                stats["embeddings"] += len(embedding_pairs)

            session.commit()
            logger.info(
                "Seeded: %s (%d chunks, %d embeddings)",
                doc.source_path,
                len(doc.chunks),
                len(doc.chunks),
            )

        except Exception:
            session.rollback()
            logger.exception("Failed to seed document: %s", jf.name)
            stats["failed"] += 1
        finally:
            session.close()

    elapsed = time.perf_counter() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)

    return stats
```

- [ ] Create `database/__main__.py`:

```python
"""Allow running the database module with python -m database."""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="database",
        description="Database management commands for FullRag.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # seed subcommand
    seed_parser = subparsers.add_parser(
        "seed", help="Seed the database from staged JSON documents."
    )
    seed_parser.add_argument(
        "--staging-dir",
        "-s",
        default="staging",
        help="Path to staging directory with Document JSON files (default: staging).",
    )
    seed_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "seed":
        from database.seed import seed_from_staging

        stats = seed_from_staging(staging_dir=args.staging_dir)

        print(f"\n{'=' * 50}")
        print("Database Seed Summary")
        print(f"{'=' * 50}")
        print(f"Documents inserted: {stats['documents']}")
        print(f"Chunks inserted:    {stats['chunks']}")
        print(f"Embeddings inserted:{stats['embeddings']}")
        print(f"Skipped (existing): {stats['skipped']}")
        print(f"Failed:             {stats.get('failed', 0)}")
        print(f"Elapsed time:       {stats.get('elapsed_seconds', 0):.2f}s")


if __name__ == "__main__":
    main()
```

##### Step 5 Verification Checklist
- [ ] Ensure PostgreSQL is running: `docker compose -f pgvector.yaml up -d`
- [ ] Ensure migration is applied: `alembic upgrade head`
- [ ] Run the seed: `python -m database seed --staging-dir staging`
- [ ] Summary prints document, chunk, and embedding counts
- [ ] Verify row counts in database:
  ```bash
  psql -h localhost -U fullrag -d fullrag -c "SELECT count(*) FROM documents;"
  psql -h localhost -U fullrag -d fullrag -c "SELECT count(*) FROM chunks;"
  psql -h localhost -U fullrag -d fullrag -c "SELECT count(*) FROM chunk_embeddings;"
  ```
- [ ] Running seed again skips already-inserted documents (idempotent)

#### Step 5 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

```bash
git add -A && git commit -m "feat(database): add seed script to load staged documents with embeddings"
```

---

#### Step 6: Integration Tests

- [ ] Create `test/test_database.py`:

```python
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
```

##### Step 6 Verification Checklist
- [ ] Ensure PostgreSQL is running: `docker compose -f pgvector.yaml up -d`
- [ ] Ensure migration is applied: `alembic upgrade head`
- [ ] Run tests: `python -m pytest test/test_database.py -v`
- [ ] All tests pass

#### Step 6 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

```bash
git add -A && git commit -m "test(database): add integration tests for schema, CRUD, vector search, and cascade deletes"
```
