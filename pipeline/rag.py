"""RAG pipeline orchestrator — wires retrieval, context, prompts, and generation."""

import logging
import time

from generation.context_manager import ContextManager
from generation.llm_service import GenerationProvider, get_generation_provider
from generation.models import GenerationConfig, PromptVariant
from generation.prompts import PromptRegistry
from pipeline.models import (
    LatencyBreakdown,
    RAGResponse,
    SourceCitation,
)
from retrieval.models import RetrievalResult
from retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve -> context -> prompt -> generate.

    Orchestrates the full flow from question to grounded answer.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_provider: GenerationProvider | None = None,
        context_manager: ContextManager | None = None,
        prompt_registry: PromptRegistry | None = None,
        generation_config: GenerationConfig | None = None,
        provider_name: str = "gemini",
    ) -> None:
        self._retrieval = retrieval_service or RetrievalService()
        self._generation_config = generation_config or GenerationConfig()
        self._provider = generation_provider or get_generation_provider(provider_name)
        self._context_manager = context_manager or ContextManager(self._generation_config)
        self._prompt_registry = prompt_registry or PromptRegistry()

    def query(
        self,
        question: str,
        top_k: int = 5,
        prompt_variant: str | None = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline for a question.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            prompt_variant: Explicit prompt variant name, or None for auto-select.

        Returns:
            RAGResponse with answer, sources, latency, and token usage.
        """
        total_start = time.perf_counter()
        latency = LatencyBreakdown()

        # Step 1: Retrieve relevant chunks
        logger.info("RAG pipeline: retrieving for '%s' (top_k=%d)", question[:80], top_k)
        retrieval_start = time.perf_counter()
        retrieval_response = self._retrieval.retrieve(question, top_k=top_k)
        latency.retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        # Step 2: Fit context to token budget
        context_start = time.perf_counter()
        fitted_chunks = self._context_manager.fit_context(retrieval_response.results)
        latency.context_ms = round((time.perf_counter() - context_start) * 1000, 2)

        # Step 3: Select prompt template
        if prompt_variant:
            variant = PromptVariant(prompt_variant)
        else:
            variant = self._prompt_registry.select_template(question, fitted_chunks)

        # Step 4: Render prompt
        rendered = self._prompt_registry.render(variant, question, fitted_chunks)

        # Step 5: Generate answer
        logger.info(
            "RAG pipeline: generating with variant=%s, version=%s",
            rendered.variant.value,
            rendered.version,
        )
        generation_start = time.perf_counter()
        gen_response = self._provider.generate(
            rendered.system_prompt,
            rendered.user_prompt,
            self._generation_config,
        )
        latency.generation_ms = round((time.perf_counter() - generation_start) * 1000, 2)

        # Step 6: Package response
        latency.total_ms = round((time.perf_counter() - total_start) * 1000, 2)

        sources = self._build_citations(fitted_chunks)

        logger.info(
            "RAG pipeline complete: answer_len=%d, sources=%d, total_ms=%.1f",
            len(gen_response.text),
            len(sources),
            latency.total_ms,
        )

        return RAGResponse(
            answer=gen_response.text,
            sources=sources,
            latency=latency,
            token_usage=gen_response.token_usage,
            prompt_version=rendered.version,
        )

    @staticmethod
    def _build_citations(chunks: list[RetrievalResult]) -> list[SourceCitation]:
        """Convert retrieval results to source citations."""
        return [
            SourceCitation(
                document_title=chunk.document_title,
                source_path=chunk.source_path,
                chunk_summary=chunk.summary,
                page_numbers=chunk.page_numbers,
                similarity_score=chunk.similarity_score,
            )
            for chunk in chunks
        ]
