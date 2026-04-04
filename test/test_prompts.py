"""Tests for prompt templates and registry."""

import pytest

from generation.models import PromptVariant, RenderedPrompt
from generation.prompt_templates import insufficient_v1, qa_v1, summarize_v1
from generation.prompts import PromptRegistry
from retrieval.models import RetrievalResult


def _make_result(score: float = 0.85, text: str = "Sample chunk text.") -> RetrievalResult:
    """Create a minimal RetrievalResult for testing."""
    return RetrievalResult(
        chunk_id="c-1",
        text=text,
        summary="A summary.",
        keywords=["test"],
        section_path=["Section 1"],
        page_numbers=[1, 2],
        document_id="d-1",
        document_title="Test Document",
        source_path="test/doc.md",
        similarity_score=score,
        token_count=10,
    )


class TestQATemplate:
    """Tests for the Q&A prompt template."""

    def test_render_returns_rendered_prompt(self):
        chunks = [_make_result()]
        result = qa_v1.render("What is X?", chunks)
        assert isinstance(result, RenderedPrompt)
        assert result.variant == PromptVariant.QA
        assert result.version == "qa_v1"

    def test_system_prompt_contains_grounding_rules(self):
        result = qa_v1.render("Q?", [_make_result()])
        assert "ONLY based on" in result.system_prompt
        assert "Source" in result.system_prompt

    def test_user_prompt_contains_query(self):
        result = qa_v1.render("What is polymers?", [_make_result()])
        assert "What is polymers?" in result.user_prompt

    def test_user_prompt_contains_context_with_attribution(self):
        chunks = [
            _make_result(text="Polymers are long chain molecules."),
        ]
        result = qa_v1.render("Q?", chunks)
        assert "Test Document" in result.user_prompt
        assert "Page 1, 2" in result.user_prompt
        assert "Polymers are long chain molecules." in result.user_prompt

    def test_multiple_chunks_all_present(self):
        chunks = [_make_result(text=f"Chunk {i}") for i in range(3)]
        result = qa_v1.render("Q?", chunks)
        for i in range(3):
            assert f"Chunk {i}" in result.user_prompt


class TestSummarizeTemplate:
    """Tests for the summarization prompt template."""

    def test_render_returns_summarize_variant(self):
        result = summarize_v1.render("Summarize X.", [_make_result()])
        assert result.variant == PromptVariant.SUMMARIZE
        assert result.version == "summarize_v1"

    def test_system_prompt_contains_summarization_instruction(self):
        result = summarize_v1.render("Summarize.", [_make_result()])
        assert "summary" in result.system_prompt.lower()


class TestInsufficientTemplate:
    """Tests for the insufficient context template."""

    def test_render_with_low_confidence_chunks(self):
        chunks = [_make_result(score=0.1)]
        result = insufficient_v1.render("Unknown topic?", chunks)
        assert result.variant == PromptVariant.INSUFFICIENT
        assert (
            "low relevance" in result.user_prompt.lower()
            or "low confidence" in result.user_prompt.lower()
        )

    def test_render_with_no_chunks(self):
        result = insufficient_v1.render("Unknown?", [])
        assert "No relevant documents" in result.user_prompt


class TestPromptRegistry:
    """Tests for the PromptRegistry auto-selection logic."""

    def test_selects_qa_for_high_scores(self):
        registry = PromptRegistry()
        results = [_make_result(score=0.85)]
        variant = registry.select_template("Q?", results)
        assert variant == PromptVariant.QA

    def test_selects_insufficient_for_low_scores(self):
        registry = PromptRegistry(insufficient_threshold=0.3)
        results = [_make_result(score=0.1)]
        variant = registry.select_template("Q?", results)
        assert variant == PromptVariant.INSUFFICIENT

    def test_selects_insufficient_for_empty_results(self):
        registry = PromptRegistry()
        variant = registry.select_template("Q?", [])
        assert variant == PromptVariant.INSUFFICIENT

    def test_render_qa(self):
        registry = PromptRegistry()
        rendered = registry.render(PromptVariant.QA, "Q?", [_make_result()])
        assert isinstance(rendered, RenderedPrompt)
        assert rendered.variant == PromptVariant.QA

    def test_render_summarize(self):
        registry = PromptRegistry()
        rendered = registry.render(
            PromptVariant.SUMMARIZE, "Summarize.", [_make_result()]
        )
        assert rendered.variant == PromptVariant.SUMMARIZE

    def test_render_insufficient(self):
        registry = PromptRegistry()
        rendered = registry.render(PromptVariant.INSUFFICIENT, "Q?", [])
        assert rendered.variant == PromptVariant.INSUFFICIENT

    def test_custom_threshold(self):
        registry = PromptRegistry(insufficient_threshold=0.7)
        results = [_make_result(score=0.5)]
        variant = registry.select_template("Q?", results)
        assert variant == PromptVariant.INSUFFICIENT
