"""LLM service with Gemini provider for chunk enrichment."""

import json
import logging
import time

from google import genai
from google.genai import types

from config.settings import get_gemini_api_key
from generation.models import ChunkEnrichment, LLMConfig

logger = logging.getLogger(__name__)


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
