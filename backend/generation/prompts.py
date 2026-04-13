"""Prompt templates and registry for LLM-driven enrichment and generation."""

import logging
from typing import Callable

from generation.models import PromptVariant, RenderedPrompt
from generation.prompt_templates import insufficient_v1, qa_v1, summarize_v1, transcript_fallback_v1
from retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Existing enrichment prompt (unchanged)
# ---------------------------------------------------------------------------

ENRICHMENT_SYSTEM_PROMPT = """\
You are a metadata extraction assistant for a document retrieval system.

Given a chunk of text from a document, produce structured metadata to improve search and retrieval quality.

Rules:
1. **Summary**: Write a concise 1-2 sentence summary of what the chunk contains. \
Only describe information that is explicitly present in the text. Never add information \
that is not in the chunk.

2. **Keywords**: Extract 3 to 7 specific, domain-relevant keywords or key phrases. \
Do NOT include generic stopwords (e.g., "the", "and", "information"). \
Prefer proper nouns, technical terms, and specific concepts over generic words.

3. **Hypothetical Questions**: Generate 2 to 5 questions that a user might ask \
which this chunk could answer. Write them as natural questions a real user would type \
into a search bar. Each question should be answerable using only the content in this chunk.

Output your response as JSON conforming to the provided schema.\
"""

# ---------------------------------------------------------------------------
# Prompt Registry for generation
# ---------------------------------------------------------------------------

# Type alias for template render functions
TemplateRenderer = Callable[[str, list[RetrievalResult]], RenderedPrompt]

# Default similarity threshold below which we consider context insufficient
DEFAULT_INSUFFICIENT_THRESHOLD = 0.3


class PromptRegistry:
	"""Registry mapping PromptVariant to template renderers.

	Supports auto-selection based on retrieval similarity scores.
	"""

	def __init__(
		self,
		insufficient_threshold: float = DEFAULT_INSUFFICIENT_THRESHOLD,
	) -> None:
		self._insufficient_threshold = insufficient_threshold
		self._templates: dict[PromptVariant, TemplateRenderer] = {
			PromptVariant.QA: qa_v1.render,
			PromptVariant.SUMMARIZE: summarize_v1.render,
			PromptVariant.INSUFFICIENT: insufficient_v1.render,
			PromptVariant.TRANSCRIPT_FALLBACK: transcript_fallback_v1.render,
		}

	def select_template(
		self,
		query: str,
		retrieval_results: list[RetrievalResult],
	) -> PromptVariant:
		"""Auto-select the best prompt variant based on retrieval quality.

		If all similarity scores are below the threshold, selects INSUFFICIENT.
		Otherwise selects QA (default).

		Args:
			query: The user's query (reserved for future heuristics).
			retrieval_results: Retrieved chunks with similarity scores.

		Returns:
			The selected PromptVariant.
		"""
		if not retrieval_results:
			logger.info("No retrieval results — selecting INSUFFICIENT template")
			return PromptVariant.INSUFFICIENT

		max_score = max(r.similarity_score for r in retrieval_results)
		if max_score < self._insufficient_threshold:
			logger.info(
				"Max similarity %.4f < threshold %.4f — selecting INSUFFICIENT",
				max_score,
				self._insufficient_threshold,
			)
			return PromptVariant.INSUFFICIENT

		logger.info("Max similarity %.4f — selecting QA template", max_score)
		return PromptVariant.QA

	def render(
		self,
		variant: PromptVariant,
		query: str,
		context_chunks: list[RetrievalResult],
	) -> RenderedPrompt:
		"""Render a prompt using the specified variant template.

		Args:
			variant: Which prompt template to use.
			query: The user's query.
			context_chunks: Retrieved context chunks.

		Returns:
			RenderedPrompt with system and user prompts.

		Raises:
			KeyError: If the variant is not registered.
		"""
		renderer = self._templates[variant]
		rendered = renderer(query, context_chunks)
		logger.info(
			"Rendered prompt: variant=%s, version=%s",
			rendered.variant.value,
			rendered.version,
		)
		return rendered
