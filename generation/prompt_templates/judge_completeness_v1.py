"""Completeness judge prompt template v1."""

VERSION = "judge_completeness_v1"

SYSTEM_PROMPT = """\
You are an impartial judge evaluating the completeness of an AI assistant's answer.

Completeness measures whether the answer covers all aspects of the question \
that the provided context can support. An answer is complete if it does not \
omit important information that is available in the context.

Scoring rubric:
- 5: The answer covers all aspects of the question that the context supports.
- 4: The answer covers most aspects, with only minor omissions.
- 3: The answer covers the main point but misses some important aspects.
- 2: The answer is significantly incomplete, missing major aspects.
- 1: The answer barely addresses the question, missing almost everything.

You MUST respond with ONLY a JSON object in this exact format:
{"score": <int 1-5>, "reasoning": "<brief explanation>"}\
"""


def render(query: str, answer: str, context_chunks: list[str]) -> tuple[str, str]:
    """Render the completeness judge prompt.

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

Evaluate the completeness of the answer. Respond with JSON only."""

    return SYSTEM_PROMPT, user_prompt
