"""Data models for the generation service."""

from enum import Enum

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


class GenerationConfig(BaseModel):
    """Configuration for answer generation."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 2048
    max_context_tokens: int = 6000
    context_budget_ratio: float = 0.75


class TokenUsage(BaseModel):
    """Token usage from a generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GenerationResponse(BaseModel):
    """Response from an LLM generation call."""

    text: str = Field(description="Generated answer text.")
    model: str = Field(description="Model name used for generation.")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str = Field(default="stop", description="Why generation stopped.")


class PromptVariant(str, Enum):
    """Supported prompt template variants."""

    QA = "qa"
    SUMMARIZE = "summarize"
    INSUFFICIENT = "insufficient"
    TRANSCRIPT_FALLBACK = "transcript_fallback"


class RenderedPrompt(BaseModel):
    """A fully rendered prompt ready for LLM submission."""

    system_prompt: str
    user_prompt: str
    variant: PromptVariant
    version: str
