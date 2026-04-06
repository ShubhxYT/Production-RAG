"""Retrieval service orchestrating query embedding and vector search."""

import asyncio
import time

from observability.logging import get_logger

from database.connection import get_session
from database.models import ChunkModel
from database.repository import DocumentRepository
from embeddings.cache import CachedEmbeddingService
from embeddings.models import EmbeddingConfig
from embeddings.service import EmbeddingService
from retrieval.models import RetrievalResponse, RetrievalResult

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates query embedding -> vector search -> result formatting.

    Uses the local SentenceTransformer embedding model and pgvector
    cosine similarity search via DocumentRepository.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | CachedEmbeddingService | None = None,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize the retrieval service.

        Args:
            embedding_service: Pre-configured embedding service. If None,
                a default EmbeddingService is created using the config.
            config: Embedding configuration. Used only when
                embedding_service is not provided.
        """
        self._config = config or EmbeddingConfig()
        self._embedding_service = embedding_service or EmbeddingService(
            config=self._config,
        )
        self._repo = DocumentRepository()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> RetrievalResponse:
        """Retrieve the most relevant chunks for a query.

        Embeds the query, searches the vector database for similar chunks,
        and returns ranked results with metadata.

        Args:
            query: The user's search query.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0-1). None disables filtering.

        Returns:
            RetrievalResponse with ranked results and query metadata.
        """
        start = time.perf_counter()

        # 1. Embed the query (blocking I/O — offload to thread)
        logger.debug(
            "Embedding query",
            extra={"component": "retrieval", "query": query[:80]},
        )
        query_vector = await asyncio.to_thread(
            self._embedding_service.embed_one, query
        )

        # 2. Search the vector database (blocking I/O — offload to thread)
        results = await asyncio.to_thread(
            self._search_sync, query_vector, top_k, threshold
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        top_score = results[0].similarity_score if results else 0.0
        logger.info(
            "Retrieval complete",
            extra={
                "component": "retrieval",
                "result_count": len(results),
                "duration_ms": round(elapsed_ms, 2),
                "top_k": top_k,
                "threshold": threshold,
                "top_score": top_score,
            },
        )

        return RetrievalResponse(
            query=query,
            top_k=top_k,
            threshold=threshold,
            result_count=len(results),
            latency_ms=round(elapsed_ms, 2),
            results=results,
        )

    def retrieve_sync(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> RetrievalResponse:
        """Synchronous retrieve for CLI and backward compatibility."""
        start = time.perf_counter()

        query_vector = self._embedding_service.embed_one(query)
        results = self._search_sync(query_vector, top_k, threshold)

        elapsed_ms = (time.perf_counter() - start) * 1000
        top_score = results[0].similarity_score if results else 0.0
        logger.info(
            "Retrieval complete",
            extra={
                "component": "retrieval",
                "result_count": len(results),
                "duration_ms": round(elapsed_ms, 2),
                "top_k": top_k,
                "threshold": threshold,
                "top_score": top_score,
            },
        )

        return RetrievalResponse(
            query=query,
            top_k=top_k,
            threshold=threshold,
            result_count=len(results),
            latency_ms=round(elapsed_ms, 2),
            results=results,
        )

    def _search_sync(
        self,
        query_vector: list[float],
        top_k: int,
        threshold: float | None,
    ) -> list[RetrievalResult]:
        """Run the synchronous DB search and map results."""
        session = get_session()
        try:
            raw_results = self._repo.search_by_vector(
                session, query_vector, top_k=top_k, threshold=threshold,
            )

            # 3. Map to RetrievalResult objects
            results = [
                self._map_result(chunk, score) for chunk, score in raw_results
            ]
        finally:
            session.close()
        return results

    @staticmethod
    def _map_result(chunk: ChunkModel, score: float) -> RetrievalResult:
        """Map a ChunkModel + similarity score to a RetrievalResult.

        Accesses the chunk's parent document via the SQLAlchemy relationship
        to populate document-level fields.
        """
        doc = chunk.document
        return RetrievalResult(
            chunk_id=str(chunk.id),
            text=chunk.text,
            summary=chunk.summary or "",
            keywords=list(chunk.keywords or []),
            section_path=list(chunk.section_path or []),
            page_numbers=list(chunk.page_numbers or []),
            document_id=str(chunk.document_id),
            document_title=doc.title if doc else None,
            source_path=doc.source_path if doc else "",
            similarity_score=round(score, 4),
            token_count=chunk.token_count or 0,
        )