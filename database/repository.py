"""Data access layer for documents, chunks, and embeddings."""

import hashlib
import logging

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from database.models import ChunkEmbeddingModel, ChunkModel, DocumentModel, QueryLogModel
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

    def search_by_keyword(
        self,
        session: Session,
        query: str,
        limit: int = 10,
    ) -> list[tuple[ChunkModel, float]]:
        """Full-text search over chunk text using the tsv GIN index.

        Uses plainto_tsquery to safely parse arbitrary user input.
        Returns ts_rank scores normalized to [0, 1].

        Args:
            session: Active SQLAlchemy session.
            query: User's search query.
            limit: Maximum number of results to return.

        Returns:
            List of (ChunkModel, score) tuples, highest score first.
        """
        if not query or not query.strip():
            return []

        stmt = text("""
            SELECT c.id AS chunk_id,
                   ts_rank(c.tsv, plainto_tsquery('english', :q)) AS score
            FROM chunks c
            WHERE c.tsv @@ plainto_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT :lim
        """)

        rows = session.execute(stmt, {"q": query, "lim": limit}).all()

        if not rows:
            return []

        # Normalize scores to [0, 1] by dividing by max score
        max_score = max(r.score for r in rows)
        if max_score <= 0:
            return []

        chunk_ids = [str(r.chunk_id) for r in rows]
        chunks_stmt = select(ChunkModel).where(ChunkModel.id.in_(chunk_ids))
        chunks_by_id = {
            str(c.id): c for c in session.execute(chunks_stmt).scalars().all()
        }

        results: list[tuple[ChunkModel, float]] = []
        for row in rows:
            chunk = chunks_by_id.get(str(row.chunk_id))
            if chunk:
                normalized_score = float(row.score) / max_score
                results.append((chunk, normalized_score))

        return results

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
            stmt = stmt.where(ChunkModel.keywords.op("&&")(keywords))

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

    def insert_query_log(
        self,
        session: Session,
        log_data: dict,
    ) -> str:
        """Insert a query audit log record.

        Args:
            session: Active SQLAlchemy session.
            log_data: Dictionary with query log fields.

        Returns:
            The query log record ID.
        """
        record = QueryLogModel(**log_data)
        session.add(record)
        session.flush()
        logger.debug("Inserted query log %s", record.id)
        return record.id
