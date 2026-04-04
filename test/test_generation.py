"""Tests for the generation service and context manager."""

from unittest.mock import MagicMock, patch

import pytest

from generation.context_manager import ContextManager
from generation.llm_service import (
    CerebrasProvider,
    GeminiGenerationProvider,
    get_generation_provider,
)
from generation.models import (
    GenerationConfig,
    GenerationResponse,
)
from retrieval.models import RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, score: float = 0.9) -> RetrievalResult:
    """Create a minimal RetrievalResult for testing."""
    return RetrievalResult(
        chunk_id="chunk-1",
        text=text,
        summary="A summary.",
        keywords=["test"],
        section_path=[],
        page_numbers=[1],
        document_id="doc-1",
        document_title="Test Doc",
        source_path="test/doc.md",
        similarity_score=score,
        token_count=0,
    )


# ---------------------------------------------------------------------------
# ContextManager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    """Unit tests for ContextManager."""

    def test_count_tokens_simple(self):
        cm = ContextManager()
        count = cm.count_tokens("Hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_string(self):
        cm = ContextManager()
        assert cm.count_tokens("") == 0

    def test_count_tokens_known_value(self):
        cm = ContextManager()
        # "Hello" is typically 1 token in cl100k_base
        count = cm.count_tokens("Hello")
        assert count >= 1

    def test_fit_context_selects_all_when_budget_allows(self):
        cm = ContextManager(GenerationConfig(max_context_tokens=10000))
        chunks = [_make_chunk("Short text.") for _ in range(3)]
        selected = cm.fit_context(chunks)
        assert len(selected) == 3

    def test_fit_context_truncates_when_over_budget(self):
        cm = ContextManager(
            GenerationConfig(max_context_tokens=10, context_budget_ratio=1.0)
        )
        # Each chunk has ~50 tokens, budget is only 10
        chunks = [_make_chunk("word " * 50) for _ in range(3)]
        selected = cm.fit_context(chunks)
        assert len(selected) < 3

    def test_fit_context_empty_list(self):
        cm = ContextManager()
        assert cm.fit_context([]) == []

    def test_fit_context_respects_explicit_max_tokens(self):
        cm = ContextManager(GenerationConfig(max_context_tokens=100000))
        chunks = [_make_chunk("word " * 50) for _ in range(5)]
        # Set very low explicit max
        selected = cm.fit_context(chunks, max_tokens=5)
        assert len(selected) < 5

    def test_fit_context_preserves_order(self):
        cm = ContextManager(GenerationConfig(max_context_tokens=10000))
        chunks = [
            _make_chunk(f"Chunk {i} content.", score=0.9 - i * 0.1)
            for i in range(3)
        ]
        selected = cm.fit_context(chunks)
        for i, chunk in enumerate(selected):
            assert chunk.text == f"Chunk {i} content."


# ---------------------------------------------------------------------------
# Generation provider tests
# ---------------------------------------------------------------------------


class TestGetGenerationProvider:
    """Tests for the provider factory."""

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_generation_provider("nonexistent")

    @patch("generation.llm_service.get_gemini_api_key", return_value="fake-key")
    @patch("generation.llm_service.genai")
    def test_gemini_provider_instantiation(self, mock_genai, mock_key):
        provider = get_generation_provider("gemini")
        assert isinstance(provider, GeminiGenerationProvider)

    @patch("generation.llm_service.get_cerebras_api_key", return_value="fake-key")
    @patch(
        "generation.llm_service.get_cerebras_base_url", return_value="https://fake.api"
    )
    def test_cerebras_provider_instantiation(self, mock_url, mock_key):
        provider = get_generation_provider("cerebras")
        assert isinstance(provider, CerebrasProvider)


class TestGeminiGenerationProvider:
    """Tests for GeminiGenerationProvider with mocked API."""

    @patch("generation.llm_service.get_gemini_api_key", return_value="fake-key")
    @patch("generation.llm_service.genai")
    def test_generate_returns_response(self, mock_genai, mock_key):
        # Set up mock
        mock_response = MagicMock()
        mock_response.text = "This is the generated answer."
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiGenerationProvider()
        config = GenerationConfig()
        result = provider.generate("system", "user query", config)

        assert isinstance(result, GenerationResponse)
        assert result.text == "This is the generated answer."
        assert result.token_usage.prompt_tokens == 100
        assert result.token_usage.completion_tokens == 50

    @patch("generation.llm_service.get_gemini_api_key", return_value="fake-key")
    @patch("generation.llm_service.genai")
    def test_generate_retries_on_failure(self, mock_genai, mock_key):
        mock_response = MagicMock()
        mock_response.text = "Answer after retry."
        mock_response.usage_metadata = None

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            ValueError("Transient error"),
            mock_response,
        ]
        mock_genai.Client.return_value = mock_client

        provider = GeminiGenerationProvider()
        config = GenerationConfig()

        with patch("generation.llm_service.time.sleep"):
            result = provider.generate("system", "user query", config)

        assert result.text == "Answer after retry."
        assert mock_client.models.generate_content.call_count == 2


class TestCerebrasProvider:
    """Tests for CerebrasProvider with mocked OpenAI client."""

    @patch("generation.llm_service.get_cerebras_api_key", return_value="fake-key")
    @patch(
        "generation.llm_service.get_cerebras_base_url", return_value="https://fake.api"
    )
    def test_generate_returns_response(self, mock_url, mock_key):
        provider = CerebrasProvider()

        # Mock the OpenAI client's create method
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Cerebras answer."
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 40
        mock_response.usage.total_tokens = 120

        provider._client.chat.completions.create = MagicMock(return_value=mock_response)

        config = GenerationConfig(model_name="llama-3.3-70b")
        result = provider.generate("system", "user query", config)

        assert isinstance(result, GenerationResponse)
        assert result.text == "Cerebras answer."
        assert result.token_usage.total_tokens == 120
