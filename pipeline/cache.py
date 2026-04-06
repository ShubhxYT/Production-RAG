"""In-memory response cache with TTL expiry for the RAG pipeline."""

import hashlib
import logging

from cachetools import TTLCache

from pipeline.models import RAGResponse

logger = logging.getLogger(__name__)


def _query_hash(query: str) -> str:
    """SHA-256 hash of the query text for use as cache key."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


class ResponseCache:
    """TTL-based in-memory cache for RAGResponse objects.

    Stores serialized responses keyed by SHA-256(query_text).
    Expired entries are evicted automatically by cachetools.
    """

    def __init__(self, maxsize: int = 256, ttl: int = 3600) -> None:
        self._cache: TTLCache[str, dict] = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, query: str) -> RAGResponse | None:
        """Look up a cached response.

        Args:
            query: The raw query text.

        Returns:
            Cached RAGResponse or None on miss.
        """
        key = _query_hash(query)
        data = self._cache.get(key)
        if data is not None:
            logger.debug("Response cache HIT for query hash %s", key[:12])
            return RAGResponse.model_validate(data)
        return None

    def set(self, query: str, response: RAGResponse) -> None:
        """Store a response in the cache.

        Args:
            query: The raw query text.
            response: The RAGResponse to cache.
        """
        key = _query_hash(query)
        self._cache[key] = response.model_dump()
        logger.debug("Response cache SET for query hash %s", key[:12])

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        logger.info("Response cache invalidated")

    @property
    def size(self) -> int:
        """Current number of (non-expired) entries."""
        return self._cache.currsize
