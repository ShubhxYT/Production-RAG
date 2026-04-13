"""Pydantic models for the RAG pipeline."""

from pydantic import BaseModel, Field

from generation.models import TokenUsage


class SourceCitation(BaseModel):
    """A source citation for a RAG response."""

    document_title: str | None = Field(
        default=None, description="Title of the source document."
    )
    source_path: str = Field(description="File path of the source document.")
    chunk_summary: str = Field(description="Summary of the cited chunk.")
    page_numbers: list[int] = Field(
        default_factory=list, description="Page numbers the chunk spans."
    )
    similarity_score: float = Field(description="Cosine similarity score.")


class LatencyBreakdown(BaseModel):
    """Latency breakdown for each pipeline stage."""

    retrieval_ms: float = 0.0
    context_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class RAGRequest(BaseModel):
    """Request to the RAG pipeline."""

    question: str = Field(
        description="The user's question.",
        min_length=1,
        max_length=1000,
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    prompt_variant: str | None = Field(
        default=None,
        description="Explicit prompt variant (qa, summarize). Auto-selects if None.",
    )


class RAGResponse(BaseModel):
    """Response from the RAG pipeline."""

    answer: str = Field(description="Generated answer text.")
    sources: list[SourceCitation] = Field(
        default_factory=list, description="Source citations."
    )
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    prompt_version: str = Field(
        default="", description="Which prompt template version was used."
    )
