"""Unit tests for the embedding service."""

import tempfile
from pathlib import Path

from embeddings.cache import CachedEmbeddingService, _cache_key
from embeddings.models import EmbeddingConfig, EmbeddingResult
from embeddings.service import EmbeddingService


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_embedding_config_defaults():
    config = EmbeddingConfig()
    assert config.model_name == "BAAI/bge-base-en-v1.5"
    assert config.dimensions == 768
    assert config.batch_size == 100
    assert config.max_retries == 3
    assert config.cache_dir is None


def test_embedding_result_roundtrip():
    result = EmbeddingResult(
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        model="test-model",
        dimensions=2,
        token_usage=0,
    )
    data = result.model_dump()
    restored = EmbeddingResult.model_validate(data)
    assert restored.vectors == result.vectors
    assert restored.token_usage == 0


# ---------------------------------------------------------------------------
# Provider mock
# ---------------------------------------------------------------------------


class MockProvider:
    """Mock embedding provider for testing (no GPU or model loading required)."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.call_count = 0
        self.last_texts: list[str] = []

    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResult:
        self.call_count += 1
        self.last_texts = texts
        vectors = [[float(i)] * self.dim for i in range(len(texts))]
        return EmbeddingResult(
            vectors=vectors,
            model=config.model_name,
            dimensions=self.dim,
            token_usage=0,  # Local models have no API token cost
        )


# ---------------------------------------------------------------------------
# EmbeddingService - batching
# ---------------------------------------------------------------------------


def test_embed_empty():
    provider = MockProvider()
    config = EmbeddingConfig(batch_size=10)
    service = EmbeddingService(provider=provider, config=config)
    result = service.embed([])
    assert result.vectors == []
    assert result.token_usage == 0
    assert provider.call_count == 0


def test_embed_single_batch():
    provider = MockProvider()
    config = EmbeddingConfig(batch_size=10)
    service = EmbeddingService(provider=provider, config=config)
    texts = [f"text {i}" for i in range(5)]
    result = service.embed(texts)
    assert len(result.vectors) == 5
    assert provider.call_count == 1


def test_embed_multiple_batches():
    provider = MockProvider()
    config = EmbeddingConfig(batch_size=3)
    service = EmbeddingService(provider=provider, config=config)
    texts = [f"text {i}" for i in range(7)]
    result = service.embed(texts)
    assert len(result.vectors) == 7
    # 7 texts / batch_size 3 = 3 batches (3, 3, 1)
    assert provider.call_count == 3


def test_embed_exact_batch_boundary():
    provider = MockProvider()
    config = EmbeddingConfig(batch_size=5)
    service = EmbeddingService(provider=provider, config=config)
    texts = [f"text {i}" for i in range(10)]
    result = service.embed(texts)
    assert len(result.vectors) == 10
    assert provider.call_count == 2


def test_embed_250_texts_splits_into_3_batches():
    provider = MockProvider()
    config = EmbeddingConfig(batch_size=100)
    service = EmbeddingService(provider=provider, config=config)
    texts = [f"text {i}" for i in range(250)]
    result = service.embed(texts)
    assert len(result.vectors) == 250
    assert provider.call_count == 3


def test_embed_token_usage_zero_for_local_model():
    provider = MockProvider()
    config = EmbeddingConfig(batch_size=2)
    service = EmbeddingService(provider=provider, config=config)
    texts = ["a", "b", "c"]
    result = service.embed(texts)
    # Local models have no API token cost
    assert result.token_usage == 0


def test_embed_one():
    provider = MockProvider(dim=3)
    config = EmbeddingConfig()
    service = EmbeddingService(provider=provider, config=config)
    vec = service.embed_one("hello")
    assert len(vec) == 3


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    key1 = _cache_key("model-a", "hello")
    key2 = _cache_key("model-a", "hello")
    assert key1 == key2


def test_cache_key_differs_by_model():
    key1 = _cache_key("model-a", "hello")
    key2 = _cache_key("model-b", "hello")
    assert key1 != key2


def test_cache_key_differs_by_text():
    key1 = _cache_key("model-a", "hello")
    key2 = _cache_key("model-a", "world")
    assert key1 != key2


# ---------------------------------------------------------------------------
# CachedEmbeddingService
# ---------------------------------------------------------------------------


def test_cached_embed_stores_and_reuses():
    provider = MockProvider(dim=2)
    config = EmbeddingConfig(model_name="test-model")
    service = EmbeddingService(provider=provider, config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cached = CachedEmbeddingService(service=service, cache_dir=cache_dir)

        # First call - all miss
        result1 = cached.embed(["hello", "world"])
        assert len(result1.vectors) == 2
        assert provider.call_count == 1

        # Second call - all hit
        result2 = cached.embed(["hello", "world"])
        assert len(result2.vectors) == 2
        assert provider.call_count == 1  # No additional calls

        # Verify same vectors returned
        assert result1.vectors == result2.vectors


def test_cached_embed_partial_hit():
    provider = MockProvider(dim=2)
    config = EmbeddingConfig(model_name="test-model")
    service = EmbeddingService(provider=provider, config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cached = CachedEmbeddingService(service=service, cache_dir=cache_dir)

        # Cache "hello"
        cached.embed(["hello"])
        assert provider.call_count == 1

        # Now embed "hello" + "world" - partial hit
        result = cached.embed(["hello", "world"])
        assert len(result.vectors) == 2
        assert provider.call_count == 2  # Only "world" was sent


def test_cached_embed_empty():
    provider = MockProvider()
    config = EmbeddingConfig()
    service = EmbeddingService(provider=provider, config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        cached = CachedEmbeddingService(
            service=service, cache_dir=Path(tmpdir)
        )
        result = cached.embed([])
        assert result.vectors == []
        assert provider.call_count == 0


def test_cached_embed_persists_to_disk():
    provider = MockProvider(dim=2)
    config = EmbeddingConfig(model_name="test-model")
    service = EmbeddingService(provider=provider, config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # First instance - populate cache
        cached1 = CachedEmbeddingService(service=service, cache_dir=cache_dir)
        cached1.embed(["hello"])

        # Second instance - should load from disk
        cached2 = CachedEmbeddingService(service=service, cache_dir=cache_dir)
        result = cached2.embed(["hello"])
        assert len(result.vectors) == 1
        # Provider was only called once (by cached1)
        assert provider.call_count == 1


def test_cached_embed_one():
    provider = MockProvider(dim=3)
    config = EmbeddingConfig()
    service = EmbeddingService(provider=provider, config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        cached = CachedEmbeddingService(
            service=service, cache_dir=Path(tmpdir)
        )
        vec = cached.embed_one("test")
        assert len(vec) == 3
