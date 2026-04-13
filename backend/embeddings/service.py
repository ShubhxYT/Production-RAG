"""Embedding service with provider abstraction, batching, and local GPU support."""

import logging
from typing import Protocol, runtime_checkable

import torch
from sentence_transformers import SentenceTransformer

from embeddings.models import EmbeddingConfig, EmbeddingResult

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding providers."""

    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Args:
            texts: Texts to embed.
            config: Embedding configuration.

        Returns:
            EmbeddingResult with vectors and metadata.
        """
        ...


class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers on local GPU (CUDA) or CPU."""

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "SentenceTransformerProvider: device=%s", self._device
        )
        self._models: dict[str, SentenceTransformer] = {}

    def _get_model(self, model_name: str) -> SentenceTransformer:
        """Load and cache a SentenceTransformer model by name.

        Args:
            model_name: HuggingFace model name or local path.

        Returns:
            Loaded SentenceTransformer model on the configured device.
        """
        if model_name not in self._models:
            logger.info("Loading model: %s on %s", model_name, self._device)
            self._models[model_name] = SentenceTransformer(
                model_name, device=self._device
            )
        return self._models[model_name]

    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResult:
        """Generate embeddings using a local sentence-transformers model.

        Args:
            texts: Texts to embed.
            config: Embedding configuration.

        Returns:
            EmbeddingResult with vectors and metadata.
        """
        model = self._get_model(config.model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = embeddings.tolist()
        actual_dimensions = len(vectors[0]) if vectors else config.dimensions

        return EmbeddingResult(
            vectors=vectors,
            model=config.model_name,
            dimensions=actual_dimensions,
            token_usage=0,  # Local models have no API token cost
        )


class EmbeddingService:
    """Embedding service with batching logic.

    Wraps an EmbeddingProvider to add automatic batching of large
    input lists. Uses sentence-transformers on local GPU by default
    - no API key required.
    """

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()
        self.provider = provider or SentenceTransformerProvider()

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a list of texts with automatic batching.

        Args:
            texts: Texts to embed.

        Returns:
            EmbeddingResult with all vectors merged.
        """
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.config.model_name,
                dimensions=self.config.dimensions,
                token_usage=0,
            )

        all_vectors: list[list[float]] = []
        total_tokens = 0

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (
                len(texts) + self.config.batch_size - 1
            ) // self.config.batch_size

            logger.info(
                "Embedding batch %d/%d (%d texts)",
                batch_num,
                total_batches,
                len(batch),
            )

            result = self._call_provider(batch)
            all_vectors.extend(result.vectors)
            total_tokens += result.token_usage

        return EmbeddingResult(
            vectors=all_vectors,
            model=self.config.model_name,
            dimensions=self.config.dimensions,
            token_usage=total_tokens,
        )

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        result = self.embed([text])
        return result.vectors[0]

    def _call_provider(self, texts: list[str]) -> EmbeddingResult:
        """Call the provider for a single batch.

        Args:
            texts: Batch of texts to embed.

        Returns:
            EmbeddingResult from the provider.
        """
        return self.provider.embed(texts, self.config)
