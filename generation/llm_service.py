"""LLM service with multi-provider support for enrichment and generation."""

import json
import logging
import time
from typing import Protocol, runtime_checkable

from google import genai
from google.genai import types

from config.settings import (
    get_cerebras_api_key,
    get_cerebras_base_url,
    get_gemini_api_key,
)
from generation.models import (
    ChunkEnrichment,
    GenerationConfig,
    GenerationResponse,
    LLMConfig,
    TokenUsage,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class GenerationProvider(Protocol):
    """Interface for LLM generation providers."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> GenerationResponse:
        """Generate a response from the LLM.

        Args:
            system_prompt: System instruction for the LLM.
            user_prompt: User message / query with context.
            config: Generation configuration.

        Returns:
            GenerationResponse with generated text and metadata.
        """
        ...


class GeminiProvider:
    """Wraps the Google GenAI client for structured chunk enrichment.

    Uses Gemini's native JSON schema output mode to guarantee
    structured responses conforming to ChunkEnrichment.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = genai.Client(api_key=get_gemini_api_key())

    def enrich_chunk(
        self, chunk_text: str, system_prompt: str
    ) -> ChunkEnrichment:
        """Generate enrichment metadata for a single chunk.

        Args:
            chunk_text: The chunk text to enrich.
            system_prompt: System instruction for the LLM.

        Returns:
            ChunkEnrichment with summary, keywords, and questions.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=chunk_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                        response_mime_type="application/json",
                        response_json_schema=ChunkEnrichment.model_json_schema(),
                    ),
                )

                # Log token usage
                if response.usage_metadata:
                    logger.debug(
                        "Token usage: prompt=%d, completion=%d, total=%d",
                        response.usage_metadata.prompt_token_count or 0,
                        response.usage_metadata.candidates_token_count or 0,
                        response.usage_metadata.total_token_count or 0,
                    )

                # Parse structured JSON response
                raw_text = response.text
                if not raw_text:
                    raise ValueError("Empty response from Gemini")

                data = json.loads(raw_text)
                return ChunkEnrichment.model_validate(data)

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt,
                        max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Enrichment failed after {max_retries} retries: {last_error}"
        )


class GeminiGenerationProvider:
    """Gemini provider for free-text answer generation."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=get_gemini_api_key())

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> GenerationResponse:
        """Generate a free-text answer using Gemini.

        Args:
            system_prompt: System instruction for the LLM.
            user_prompt: User message with context and query.
            config: Generation configuration.

        Returns:
            GenerationResponse with generated text and token usage.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=config.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=config.temperature,
                        max_output_tokens=config.max_output_tokens,
                    ),
                )

                raw_text = response.text
                if not raw_text:
                    raise ValueError("Empty response from Gemini")

                token_usage = TokenUsage()
                if response.usage_metadata:
                    token_usage = TokenUsage(
                        prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                        completion_tokens=response.usage_metadata.candidates_token_count
                        or 0,
                        total_tokens=response.usage_metadata.total_token_count or 0,
                    )

                return GenerationResponse(
                    text=raw_text,
                    model=config.model_name,
                    token_usage=token_usage,
                    finish_reason="stop",
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "Generation attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt,
                        max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Generation failed after {max_retries} retries: {last_error}"
        )


class CerebrasProvider:
    """Cerebras provider using the OpenAI-compatible API."""

    def __init__(self) -> None:
        import openai

        self._client = openai.OpenAI(
            api_key=get_cerebras_api_key(),
            base_url=get_cerebras_base_url(),
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> GenerationResponse:
        """Generate a free-text answer using Cerebras.

        Args:
            system_prompt: System instruction for the LLM.
            user_prompt: User message with context and query.
            config: Generation configuration.

        Returns:
            GenerationResponse with generated text and token usage.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_output_tokens,
                )

                choice = response.choices[0]
                raw_text = choice.message.content or ""
                if not raw_text:
                    raise ValueError("Empty response from Cerebras")

                token_usage = TokenUsage()
                if response.usage:
                    token_usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                        total_tokens=response.usage.total_tokens or 0,
                    )

                return GenerationResponse(
                    text=raw_text,
                    model=config.model_name,
                    token_usage=token_usage,
                    finish_reason=choice.finish_reason or "stop",
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "Cerebras attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt,
                        max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Cerebras generation failed after {max_retries} retries: {last_error}"
        )


def get_generation_provider(provider_name: str = "gemini") -> GenerationProvider:
    """Factory function to get a generation provider by name.

    Args:
        provider_name: Provider name ('gemini' or 'cerebras').

    Returns:
        A GenerationProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    providers = {
        "gemini": GeminiGenerationProvider,
        "cerebras": CerebrasProvider,
    }
    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Supported: {', '.join(providers)}"
        )
    return providers[provider_name]()
