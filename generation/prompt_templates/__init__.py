"""Versioned prompt templates for RAG generation."""

from generation.prompt_templates import (
	judge_coherence_v1,
	judge_completeness_v1,
	judge_faithfulness_v1,
	judge_relevance_v1,
)

JUDGE_TEMPLATES = {
	"faithfulness": judge_faithfulness_v1,
	"relevance": judge_relevance_v1,
	"completeness": judge_completeness_v1,
	"coherence": judge_coherence_v1,
}

__all__ = [
	"JUDGE_TEMPLATES",
	"judge_coherence_v1",
	"judge_completeness_v1",
	"judge_faithfulness_v1",
	"judge_relevance_v1",
]
