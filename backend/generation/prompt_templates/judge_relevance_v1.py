"""Relevance judge prompt template v1."""

VERSION = "judge_relevance_v1"

SYSTEM_PROMPT = """\
You are an impartial judge evaluating the relevance of an AI assistant's answer.

Relevance measures whether the answer actually addresses the user's question. \
An answer is relevant if it directly responds to what was asked, rather than \
providing tangential or off-topic information.

Scoring rubric:
- 5: The answer directly and fully addresses the question.
- 4: The answer mostly addresses the question with minor tangential content.
- 3: The answer partially addresses the question but misses key aspects.
- 2: The answer is mostly tangential or addresses a different question.
- 1: The answer does not address the question at all.

You MUST respond with ONLY a JSON object in this exact format:
{"score": <int 1-5>, "reasoning": "<brief explanation>"}\
"""


def render(query: str, answer: str, context_chunks: list[str]) -> tuple[str, str]:
    """Render the relevance judge prompt.

    Args:
        query: The original user query.
        answer: The generated answer to evaluate.
        context_chunks: Context texts that were provided for generation.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    context_block = "\n\n---\n\n".join(
        f"[Context {i}]\n{chunk}" for i, chunk in enumerate(context_chunks, 1)
    )

    user_prompt = f"""\
Context provided to the assistant:
{context_block}

User question: {query}

Assistant's answer: {answer}

Evaluate the relevance of the answer to the question. Respond with JSON only."""

    return SYSTEM_PROMPT, user_prompt
