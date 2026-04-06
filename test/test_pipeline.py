"""Tests for the RAG pipeline orchestrator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from generation.models import (
    GenerationConfig,
    GenerationResponse,
    TokenUsage,
)
from pipeline.models import RAGResponse
from pipeline.rag import RAGPipeline
from retrieval.models import RetrievalResponse, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrieval_result(score: float = 0.85) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c-1",
        text="Test chunk content about polymers.",
        summary="Chunk about polymers.",
        keywords=["polymers"],
        section_path=["intro"],
        page_numbers=[1],
        document_id="d-1",
        document_title="Polymer Lecture",
        source_path="data/polymer.md",
        similarity_score=score,
        token_count=10,
    )


def _make_retrieval_response(
    query: str = "test", results: list[RetrievalResult] | None = None
) -> RetrievalResponse:
    results = results or [_make_retrieval_result()]
    return RetrievalResponse(
        query=query,
        top_k=5,
        threshold=None,
        result_count=len(results),
        latency_ms=10.0,
        results=results,
    )


def _make_generation_response(text: str = "Generated answer.") -> GenerationResponse:
    return GenerationResponse(
        text=text,
        model="gemini-2.5-flash",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        finish_reason="stop",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Tests for the RAGPipeline orchestrator."""

    def test_query_returns_rag_response(self):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_response())

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response()

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
        )

        response = asyncio.run(pipeline.query("What are polymers?"))

        assert isinstance(response, RAGResponse)
        assert response.answer == "Generated answer."
        assert len(response.sources) == 1
        assert response.sources[0].document_title == "Polymer Lecture"
        assert response.token_usage.total_tokens == 150
        assert response.prompt_version != ""

    def test_latency_breakdown_non_zero(self):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_response())

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response()

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
        )

        response = asyncio.run(pipeline.query("test"))

        assert response.latency.total_ms > 0
        assert response.latency.retrieval_ms >= 0
        assert response.latency.context_ms >= 0
        assert response.latency.generation_ms >= 0

    def test_explicit_prompt_variant(self):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_response())

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response()

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
        )

        response = asyncio.run(pipeline.query("Summarize the docs", prompt_variant="summarize"))
        assert response.prompt_version == "summarize_v1"

    def test_insufficient_context_auto_selected(self):
        low_score_result = _make_retrieval_result(score=0.1)
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(
            return_value=_make_retrieval_response(results=[low_score_result])
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response(
            text="I cannot answer this."
        )

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
        )

        response = asyncio.run(pipeline.query("Unknown topic?"))
        assert response.prompt_version == "insufficient_v1"

    def test_multiple_sources_in_citations(self):
        results = [_make_retrieval_result(score=0.9 - i * 0.1) for i in range(3)]
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(
            return_value=_make_retrieval_response(results=results)
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response()

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
        )

        response = asyncio.run(pipeline.query("test"))
        assert len(response.sources) == 3

    def test_provider_failure_raises(self):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_response())

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = RuntimeError("LLM error")

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
        )

        with pytest.raises(RuntimeError, match="LLM error"):
            asyncio.run(pipeline.query("test"))


class TestResponseCache:
    """Tests for the response cache integration in RAGPipeline."""

    def test_second_call_returns_cached_response(self):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_response())

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response()

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
        )

        # First call — runs pipeline
        response1 = asyncio.run(pipeline.query("What are polymers?"))
        assert mock_retrieval.retrieve.call_count == 1
        assert mock_provider.generate.call_count == 1

        # Second call — served from cache
        response2 = asyncio.run(pipeline.query("What are polymers?"))
        assert mock_retrieval.retrieve.call_count == 1  # NOT incremented
        assert mock_provider.generate.call_count == 1  # NOT incremented
        assert response2.answer == response1.answer

    def test_cache_ttl_expiry(self):
        from pipeline.cache import ResponseCache
        import time

        # Create cache with 1-second TTL
        cache = ResponseCache(maxsize=10, ttl=1)
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_response())

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response()

        pipeline = RAGPipeline(
            retrieval_service=mock_retrieval,
            generation_provider=mock_provider,
            generation_config=GenerationConfig(max_context_tokens=10000),
            response_cache=cache,
        )

        # First call
        asyncio.run(pipeline.query("What are polymers?"))
        assert mock_retrieval.retrieve.call_count == 1

        # Expire the cache by advancing the timer
        time.sleep(1.1)

        # Second call after TTL — re-runs pipeline
        asyncio.run(pipeline.query("What are polymers?"))
        assert mock_retrieval.retrieve.call_count == 2

    def test_cache_disabled_bypasses_cache(self):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=_make_retrieval_response())

        mock_provider = MagicMock()
        mock_provider.generate.return_value = _make_generation_response()

        with patch("pipeline.rag.get_response_cache_enabled", return_value=False):
            pipeline = RAGPipeline(
                retrieval_service=mock_retrieval,
                generation_provider=mock_provider,
                generation_config=GenerationConfig(max_context_tokens=10000),
            )

        # Two calls should both run the full pipeline
        asyncio.run(pipeline.query("What are polymers?"))
        asyncio.run(pipeline.query("What are polymers?"))
        assert mock_retrieval.retrieve.call_count == 2
        assert mock_provider.generate.call_count == 2
