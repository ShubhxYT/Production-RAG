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
