"""Data models for the generation service."""

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM-based enrichment."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 1024


class ChunkEnrichment(BaseModel):
    """Structured enrichment output from the LLM.

    Used as Gemini's response_json_schema to guarantee
    structured output without manual JSON parsing.
    """

    summary: str = Field(
        description="A concise 1-2 sentence summary of what the chunk contains."
    )
    keywords: list[str] = Field(
        description="3-7 specific, domain-relevant keywords extracted from the chunk."
    )
    hypothetical_questions: list[str] = Field(
        description="2-5 hypothetical questions that this chunk could answer."
    )
