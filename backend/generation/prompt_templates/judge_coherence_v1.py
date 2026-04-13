"""Coherence judge prompt template v1."""

VERSION = "judge_coherence_v1"

SYSTEM_PROMPT = """\
You are an impartial judge evaluating the coherence of an AI assistant's answer.

Coherence measures whether the answer is well-structured, clear, logically \
consistent, and easy to understand. Evaluate the writing quality independent \
of factual accuracy.

Scoring rubric:
- 5: Exceptionally clear, well-organized, and logically structured.
- 4: Clear and well-organized with minor structural issues.
- 3: Generally understandable but could be better organized.
- 2: Confusing structure, contradictory statements, or hard to follow.
- 1: Incoherent, disorganized, or nonsensical.

You MUST respond with ONLY a JSON object in this exact format:
{"score": <int 1-5>, "reasoning": "<brief explanation>"}\
"""


def render(query: str, answer: str, context_chunks: list[str]) -> tuple[str, str]:
    """Render the coherence judge prompt.

    Args:
        query: The original user query.
        answer: The generated answer to evaluate.
        context_chunks: Context texts that were provided for generation.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_prompt = f"""\
User question: {query}

Assistant's answer: {answer}

Evaluate the coherence of the answer. Respond with JSON only."""

    return SYSTEM_PROMPT, user_prompt
