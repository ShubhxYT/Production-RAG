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
