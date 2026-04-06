"""Pydantic models for retrieval results and responses."""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A single retrieval result from vector similarity search."""

    chunk_id: str = Field(description="UUID of the matched chunk.")
    text: str = Field(description="Full chunk text content.")
    summary: str = Field(description="LLM-generated chunk summary.")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords.")
    section_path: list[str] = Field(
        default_factory=list, description="Breadcrumb path within the document."
    )
    page_numbers: list[int] = Field(
        default_factory=list, description="Page numbers the chunk spans."
    )
    document_id: str = Field(description="UUID of the parent document.")
    document_title: str | None = Field(
        default=None, description="Title of the source document."
    )
    source_path: str = Field(description="File path of the source document.")
    similarity_score: float = Field(description="Cosine similarity (0-1).")
    match_type: str = Field(
        default="vector",
        description="How this result was found: 'vector', 'keyword', or 'hybrid'.",
    )
    token_count: int = Field(default=0, description="Token count of the chunk.")


class RetrievalResponse(BaseModel):
    """Response from a retrieval query with results and metadata."""

    query: str = Field(description="The original query text.")
    top_k: int = Field(description="Maximum number of results requested.")
    threshold: float | None = Field(
        default=None, description="Minimum similarity threshold used."
    )
    result_count: int = Field(description="Number of results returned.")
    latency_ms: float = Field(description="Query latency in milliseconds.")
    results: list[RetrievalResult] = Field(
        default_factory=list, description="Ranked retrieval results."
    )
