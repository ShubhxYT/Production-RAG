# Embedding Generation Service

**Branch:** `feat/embedding-service`
**Description:** Build a model-agnostic embedding service with local CUDA GPU support, batching, and file-based caching

## Goal
Implement an embedding pipeline (Step 4 of the RAG plan) that generates vector embeddings for all document chunks using a local sentence-transformers model on the CUDA GPU (GTX 1650 SUPER). No API key required. This unblocks vector database storage (Step 6) and metadata enrichment (Step 5).

## Prerequisites
Make sure that the user is currently on the `feat/embedding-service` branch before beginning implementation.
If not, move them to the correct branch. If the branch does not exist, create it from main.

## Implementation Steps

### Step 1: Add Dependencies

- [x] Open `pyproject.toml` and add `sentence-transformers` to the dependencies list
- [x] Copy and paste the updated dependencies section into `pyproject.toml`:

```toml
dependencies = [
    "beautifulsoup4>=4.12.0",
    "langchain-docling>=2.0.0",
    "langchain-opendataloader-pdf>=2.0.0",
    "liteparse>=1.2.1",
    "markdown-it-py>=3.0.0",
    "marker-pdf>=1.10.2",
    "markdownify>=0.13.0",
    "pydantic>=2.0.0",
    "python-docx>=1.0.0",
    "sentence-transformers>=3.0.0",
    "tiktoken>=0.7.0",
    "torch>=2.11.0",
    "torchvision>=0.26.0",
]
```

- [x] Run dependency sync:

```bash
uv sync
```

##### Step 1 Verification Checklist
- [x] `uv sync` completes without errors
- [x] `python -c "import sentence_transformers; import torch; print('CUDA:', torch.cuda.is_available())"` prints `CUDA: True`

#### Step 1 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `iter-7: add sentence-transformers dependency for local GPU embeddings`

Update README.md iteration log:
```markdown
### iter-7: Embedding Service Dependencies

- Added `sentence-transformers>=3.0.0` to `pyproject.toml`
- Enables local GPU (CUDA) embedding generation — no API key required
- `torch` was already a dependency; CUDA device is used automatically when available (GTX 1650 SUPER)
```

---

### Step 2: Create Embeddings Module with Models

- [x] Create directory `embeddings/`
- [x] Create file `embeddings/__init__.py`:

```python
"""Embedding generation service for the FullRag system."""
```

- [x] Create file `embeddings/models.py`:

```python
"""Data models for the embedding service."""

from pathlib import Path

from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding service."""

    model_name: str = "BAAI/bge-base-en-v1.5"
    dimensions: int = 768
    batch_size: int = 100
    max_retries: int = 3
    cache_dir: Path | None = None


class EmbeddingResult(BaseModel):
    """Result from an embedding operation."""

    vectors: list[list[float]]
    model: str
    dimensions: int
    token_usage: int
```

##### Step 2 Verification Checklist
- [x] `python -c "from embeddings.models import EmbeddingConfig, EmbeddingResult; print('OK')"` prints `OK`
- [x] `python -c "from embeddings.models import EmbeddingConfig; c = EmbeddingConfig(); print(c.model_name, c.dimensions)"` prints `BAAI/bge-base-en-v1.5 768`

#### Step 2 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `iter-8: add embeddings module with config and result models`

Update README.md iteration log:
```markdown
### iter-8: Embeddings Module Models

- Created `embeddings/` module with `EmbeddingConfig` and `EmbeddingResult` Pydantic models
- `EmbeddingConfig`: model_name (`BAAI/bge-base-en-v1.5`), dimensions (768), batch_size, max_retries, cache_dir
- `EmbeddingResult`: vectors, model, dimensions, token_usage
```

---

### Step 3: Implement Embedding Service with Provider Abstraction

- [x] Create file `embeddings/service.py`:

```python
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
    — no API key required.
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
```

##### Step 3 Verification Checklist
- [x] `python -c "from embeddings.service import EmbeddingService, EmbeddingProvider, SentenceTransformerProvider; print('OK')"` prints `OK`
- [x] `python -c "from embeddings.service import SentenceTransformerProvider; p = SentenceTransformerProvider(); print('device:', p._device)"` prints `device: cuda`
- [x] No lint or type errors in the file

#### Step 3 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `iter-9: implement embedding service with local SentenceTransformer provider and CUDA support`

Update README.md iteration log:
```markdown
### iter-9: Embedding Service Implementation

- Created `embeddings/service.py` with:
  - `EmbeddingProvider` protocol for provider abstraction
  - `SentenceTransformerProvider` using `sentence-transformers` on local CUDA GPU (GTX 1650 SUPER; falls back to CPU)
  - `EmbeddingService` with automatic batching (configurable batch size)
  - Model is loaded lazily and cached in memory; CUDA device auto-detected via `torch.cuda.is_available()`
  - No API key required
```

---

### Step 4: Implement Local Dev Cache

- [x] Create file `embeddings/cache.py`:

```python
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
```

##### Step 4 Verification Checklist
- [x] `python -c "from embeddings.cache import CachedEmbeddingService; print('OK')"` prints `OK`
- [x] No lint or type errors

#### Step 4 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `iter-10: add file-based embedding cache for development`

Update README.md iteration log:
```markdown
### iter-10: Embedding Cache

- Created `embeddings/cache.py` with `CachedEmbeddingService`
- File-based JSON cache keyed by SHA-256 of (model_name, text)
- Separates cached/uncached texts; only sends uncached to provider
- Cache stored at `.embedding_cache/cache.json` by default
```

---

### Step 5: Add CLI for Batch Embedding

- [ ] Create file `embeddings/cli.py`:

```python
"""CLI for batch embedding of staged document chunks."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from embeddings.cache import CachedEmbeddingService
from embeddings.models import EmbeddingConfig
from embeddings.service import EmbeddingService
from ingestion.staging import load_staged_document


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="embeddings",
        description="Generate embeddings for staged document chunks using local GPU.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to staging directory with Document JSON files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".embedding_output",
        help="Output directory for embedding results (default: .embedding_output).",
    )
    parser.add_argument(
        "--cache-dir",
        default=".embedding_cache",
        help="Cache directory for dev caching (default: .embedding_cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of texts per batch (default: 100).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model name (default: BAAI/bge-base-en-v1.5).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(
            f"Error: staging directory does not exist: {args.input}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load staged documents
    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    documents = []
    for jf in json_files:
        try:
            doc = load_staged_document(jf)
            documents.append(doc)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to load: %s", jf.name
            )

    # Collect all chunks
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    for doc in documents:
        for chunk in doc.chunks:
            chunk_ids.append(chunk.id)
            chunk_texts.append(chunk.text)

    if not chunk_texts:
        print("No chunks found in staged documents.", file=sys.stderr)
        sys.exit(1)

    # Set up embedding service
    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
    )
    service = EmbeddingService(config=config)

    if args.no_cache:
        embed_service = service
    else:
        cache_dir = Path(args.cache_dir)
        embed_service = CachedEmbeddingService(
            service=service, cache_dir=cache_dir
        )

    # Embed all chunks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.perf_counter()
    result = embed_service.embed(chunk_texts)
    elapsed = time.perf_counter() - start_time

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model": result.model,
        "dimensions": result.dimensions,
        "device": device,
        "embeddings": [
            {"chunk_id": cid, "vector": vec}
            for cid, vec in zip(chunk_ids, result.vectors)
        ],
    }

    output_file = output_dir / "embeddings.json"
    output_file.write_text(
        json.dumps(output_data, indent=2), encoding="utf-8"
    )

    # Print summary — local model, no API cost
    print(f"\n{'=' * 50}")
    print("Embedding Summary")
    print(f"{'=' * 50}")
    print(f"Documents loaded:  {len(documents)}")
    print(f"Chunks embedded:   {len(chunk_ids)}")
    print(f"Model:             {result.model}")
    print(f"Dimensions:        {result.dimensions}")
    print(f"Device:            {device}")
    print(f"API cost:          $0.00 (local model)")
    print(f"Elapsed time:      {elapsed:.2f}s")
    print(f"Output file:       {output_file}")


if __name__ == "__main__":
    main()
```

- [ ] Create file `embeddings/__main__.py`:

```python
"""Allow running the embeddings module with python -m embeddings."""

from embeddings.cli import main

main()
```

##### Step 5 Verification Checklist
- [ ] `python -m embeddings --help` prints the usage message without errors
- [ ] `python -m embeddings -i staging/ -o .embedding_output` runs end-to-end (no API key required)
- [ ] Output file `.embedding_output/embeddings.json` is created with chunk IDs and vectors
- [ ] Summary shows `Device: cuda` and `API cost: $0.00 (local model)`

#### Step 5 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `iter-11: add CLI for batch embedding of staged chunks`

Update README.md iteration log:
```markdown
### iter-11: Embedding CLI

- Created `embeddings/cli.py` and `embeddings/__main__.py`
- CLI command: `python -m embeddings -i staging/ -o .embedding_output`
- Loads staged Document JSON files, extracts chunks, embeds via local SentenceTransformer on CUDA
- Supports `--no-cache`, `--batch-size`, `--model` options
- Outputs `embeddings.json` mapping chunk IDs to vectors (plus device metadata)
- Prints summary: document count, chunk count, device, elapsed time, $0.00 API cost
- No API key required

**How to run:**

```bash
# Embed all staged chunks (uses CUDA automatically)
python -m embeddings -i staging/ -o .embedding_output

# With verbose logging and no cache
python -m embeddings -i staging/ -o .embedding_output --no-cache -v

# Use a smaller/faster model
python -m embeddings -i staging/ -o .embedding_output --model BAAI/bge-small-en-v1.5
```
```

---

### Step 6: Add Unit Tests

- [ ] Create file `test/test_embeddings.py`:

```python
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
# EmbeddingService — batching
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

        # First call — all miss
        result1 = cached.embed(["hello", "world"])
        assert len(result1.vectors) == 2
        assert provider.call_count == 1

        # Second call — all hit
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

        # Now embed "hello" + "world" — partial hit
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

        # First instance — populate cache
        cached1 = CachedEmbeddingService(service=service, cache_dir=cache_dir)
        cached1.embed(["hello"])

        # Second instance — should load from disk
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
```

##### Step 6 Verification Checklist
- [ ] `pytest test/test_embeddings.py -v` — all tests pass
- [ ] No tests require an API key or GPU (all use `MockProvider`)

#### Step 6 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `iter-12: add unit tests for embedding service and cache`

Update README.md iteration log:
```markdown
### iter-12: Embedding Unit Tests

- Created `test/test_embeddings.py` with tests for:
  - Model defaults (`BAAI/bge-base-en-v1.5`, 768 dims, zero token cost)
  - Batching logic (empty, single, multiple, exact boundary, 250→3 batches)
  - Cache key determinism and uniqueness
  - Cache hit/miss/partial-hit behavior
  - Cache persistence to disk across instances
- All tests use a `MockProvider` — no GPU or model download required

**How to run:**

```bash
pytest test/test_embeddings.py -v
```
```

---

### Step 7: Add `.embedding_cache/` and `.embedding_output/` to `.gitignore`

- [ ] Append the following to `.gitignore`:

```
# Embedding cache and output
.embedding_cache/
.embedding_output/
```

##### Step 7 Verification Checklist
- [ ] `git status` does not show `.embedding_cache/` or `.embedding_output/` as untracked

#### Step 7 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `iter-13: gitignore embedding cache and output directories`

Update README.md iteration log:
```markdown
### iter-13: Gitignore Updates

- Added `.embedding_cache/` and `.embedding_output/` to `.gitignore`
```
