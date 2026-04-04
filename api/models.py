"""Pydantic request/response schemas for the API layer."""

from pydantic import BaseModel, Field

from generation.models import TokenUsage
from pipeline.models import LatencyBreakdown, SourceCitation


class QueryRequest(BaseModel):
    """POST /query request body."""

    question: str = Field(
        description="The user's question.",
        min_length=1,
        max_length=1000,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve.",
    )
    prompt_variant: str | None = Field(
        default=None,
        description="Prompt variant: 'qa', 'summarize', or None for auto.",
    )


class QueryResponse(BaseModel):
    """POST /query response body."""

    answer: str
    sources: list[SourceCitation]
    latency: LatencyBreakdown
    token_usage: TokenUsage
    prompt_version: str


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str
    database: str


class DocumentInfo(BaseModel):
    """Single document in the GET /documents response."""

    id: str
    title: str | None
    source_path: str
    chunk_count: int
    created_at: str


class DocumentsResponse(BaseModel):
    """GET /documents response body."""

    documents: list[DocumentInfo]
    total: int


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    detail: str
