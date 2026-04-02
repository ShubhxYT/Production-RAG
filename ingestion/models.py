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


class Document(BaseModel):
    """A parsed document with structural elements and metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    title: str | None = None
    format: str
    elements: list[Element] = Field(default_factory=list)
    raw_content: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
