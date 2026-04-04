"""Token counting and context window management."""

import logging

import tiktoken

from generation.models import GenerationConfig
from retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

# Use cl100k_base encoding (GPT-4 / general purpose tokenizer)
_ENCODING = tiktoken.get_encoding("cl100k_base")


class ContextManager:
    """Manages context window budget for LLM generation.

    Counts tokens and greedily selects retrieved chunks by relevance
    score until the token budget is exhausted.
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(_ENCODING.encode(text))

    def fit_context(
        self,
        chunks: list[RetrievalResult],
        max_tokens: int | None = None,
    ) -> list[RetrievalResult]:
        """Greedily select chunks by relevance until token budget is exhausted.

        Chunks are assumed to already be sorted by similarity score
        (highest first, as returned by RetrievalService).

        Args:
            chunks: Retrieved chunks sorted by relevance.
            max_tokens: Maximum token budget. Defaults to
                config.max_context_tokens * config.context_budget_ratio.

        Returns:
            Subset of chunks that fit within the token budget.
        """
        if max_tokens is None:
            max_tokens = int(
                self.config.max_context_tokens * self.config.context_budget_ratio
            )

        selected: list[RetrievalResult] = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk.text)
            if used_tokens + chunk_tokens > max_tokens:
                logger.debug(
                    "Context budget exhausted: %d/%d tokens used, "
                    "skipping chunk with %d tokens",
                    used_tokens,
                    max_tokens,
                    chunk_tokens,
                )
                break
            selected.append(chunk)
            used_tokens += chunk_tokens

        logger.info(
            "Context selection: %d/%d chunks selected, %d/%d tokens used",
            len(selected),
            len(chunks),
            used_tokens,
            max_tokens,
        )
        return selected
