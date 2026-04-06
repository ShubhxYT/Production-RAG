"""File-based embedding cache for development use."""

import hashlib
import json
import logging
from pathlib import Path

from cachetools import LRUCache

from embeddings.models import EmbeddingConfig, EmbeddingResult
from embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class InMemoryLRUCache:
    """In-memory LRU embedding cache layered on top of file-based cache.

    Keyed on (model_name, text) SHA-256 hash. Avoids disk I/O for
    repeated queries within the same process.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        self._cache: LRUCache[str, list[float]] = LRUCache(maxsize=maxsize)

    def get(self, key: str) -> list[float] | None:
        """Return cached vector or None."""
        return self._cache.get(key)

    def set(self, key: str, vector: list[float]) -> None:
        """Store a vector in the LRU cache."""
        self._cache[key] = vector

    @property
    def hits(self) -> int:
        """Current number of items in cache."""
        return self._cache.currsize


def _cache_key(model_name: str, text: str) -> str:
    """Generate a SHA-256 cache key from model name and text.

    Args:
        model_name: Name of the embedding model.
        text: Input text.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    raw = f"{model_name}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CachedEmbeddingService:
    """Wraps an EmbeddingService with file-based caching.

    Cached embeddings are stored in a JSON file keyed by
    SHA-256 hash of (model_name, text). Only uncached texts
    are sent to the underlying provider.
    """

    def __init__(
        self,
        service: EmbeddingService,
        cache_dir: Path | None = None,
        lru_maxsize: int = 1024,
    ) -> None:
        self.service = service
        self.cache_dir = cache_dir or Path(".embedding_cache")
        self._cache_file = self.cache_dir / "cache.json"
        self._cache: dict[str, list[float]] = {}
        self._lru = InMemoryLRUCache(maxsize=lru_maxsize)
        self._load_cache()

    def _load_cache(self) -> None:
        """Load the cache from disk if it exists."""
        if self._cache_file.exists():
            try:
                data = json.loads(
                    self._cache_file.read_text(encoding="utf-8")
                )
                self._cache = data
                logger.info("Loaded %d cached embeddings", len(self._cache))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load cache, starting fresh")
                self._cache = {}

    def _save_cache(self) -> None:
        """Persist the cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_text(
            json.dumps(self._cache), encoding="utf-8"
        )

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts, using LRU cache -> file cache -> provider.

        Args:
            texts: Texts to embed.

        Returns:
            EmbeddingResult with all vectors (cached + fresh).
        """
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.service.config.model_name,
                dimensions=self.service.config.dimensions,
                token_usage=0,
            )

        model = self.service.config.model_name
        keys = [_cache_key(model, t) for t in texts]

        # Separate into: LRU-hit, file-hit, uncached
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        lru_hits = 0
        file_hits = 0

        for i, key in enumerate(keys):
            lru_vec = self._lru.get(key)
            if lru_vec is not None:
                lru_hits += 1
            elif key in self._cache:
                file_hits += 1
                self._lru.set(key, self._cache[key])
            else:
                uncached_indices.append(i)
                uncached_texts.append(texts[i])

        if lru_hits > 0 or file_hits > 0:
            logger.info(
                "Cache: %d LRU hits, %d file hits, %d misses",
                lru_hits,
                file_hits,
                len(uncached_texts),
            )

        # Embed uncached texts
        total_tokens = 0
        if uncached_texts:
            result = self.service.embed(uncached_texts)
            total_tokens = result.token_usage

            # Store new embeddings in both caches
            for idx, vec in zip(uncached_indices, result.vectors):
                self._cache[keys[idx]] = vec
                self._lru.set(keys[idx], vec)
            self._save_cache()

        # Assemble final result in original order
        vectors: list[list[float]] = []
        for i, key in enumerate(keys):
            lru_vec = self._lru.get(key)
            if lru_vec is not None:
                vectors.append(lru_vec)
            else:
                vectors.append(self._cache[key])

        return EmbeddingResult(
            vectors=vectors,
            model=model,
            dimensions=self.service.config.dimensions,
            token_usage=total_tokens,
        )

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text with caching.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        result = self.embed([text])
        return result.vectors[0]
