"""Data models for the ingestion pipeline."""

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Classification of structural elements within a document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    LIST = "list"
    IMAGE = "image"


class Element(BaseModel):
    """A single structural element extracted from a document."""

    type: ElementType
    content: str
    level: int | None = None
    metadata: dict = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    """Configuration for the structure-aware chunking algorithm."""

    target_min_tokens: int = 256
    target_max_tokens: int = 512
    overlap_tokens: int = 50
    max_table_tokens: int = 1024


class Chunk(BaseModel):
    """A chunk of text produced by the structure-aware chunker."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    token_count: int
    document_id: str
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    element_types: list[ElementType] = Field(default_factory=list)
    position: int = 0
    overlap_before: str = ""

    # Enrichment fields (populated by LLM in Step 5)
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    hypothetical_questions: list[str] = Field(default_factory=list)


class Document(BaseModel):
    """A parsed document with structural elements and metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    title: str | None = None
    format: str
    elements: list[Element] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    raw_content: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Document-level metadata (for future filtering)
    doc_date: str | None = None
    doc_version: str | None = None
