"""File-based embedding cache for development use."""

import hashlib
import json
import logging
from pathlib import Path

from embeddings.models import EmbeddingConfig, EmbeddingResult
from embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.service = service
        self.cache_dir = cache_dir or Path(".embedding_cache")
        self._cache_file = self.cache_dir / "cache.json"
        self._cache: dict[str, list[float]] = {}
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
        """Embed texts, using cached results where available.

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

        # Separate cached from uncached
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, key in enumerate(keys):
            if key not in self._cache:
                uncached_indices.append(i)
                uncached_texts.append(texts[i])

        cache_hits = len(texts) - len(uncached_texts)
        if cache_hits > 0:
            logger.info(
                "Cache: %d hits, %d misses", cache_hits, len(uncached_texts)
            )

        # Embed uncached texts
        total_tokens = 0
        if uncached_texts:
            result = self.service.embed(uncached_texts)
            total_tokens = result.token_usage

            # Store new embeddings in cache
            for idx, vec in zip(uncached_indices, result.vectors):
                self._cache[keys[idx]] = vec
            self._save_cache()

        # Assemble final result in original order
        vectors = [self._cache[key] for key in keys]

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
