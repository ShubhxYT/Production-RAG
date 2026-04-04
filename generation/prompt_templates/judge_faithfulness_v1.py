"""Faithfulness judge prompt template v1."""

VERSION = "judge_faithfulness_v1"

SYSTEM_PROMPT = """\
You are an impartial judge evaluating the faithfulness of an AI assistant's answer.

Faithfulness measures whether every claim in the answer is supported by the \
provided context. An answer is faithful if it does not contain any information \
that cannot be traced back to the context documents.

Scoring rubric:
- 5: Every claim is fully supported by the context. No fabrication.
- 4: Almost all claims are supported. Minor unsupported details that don't mislead.
- 3: Most claims are supported, but some notable claims lack evidence in the context.
- 2: Several important claims are unsupported or fabricated.
- 1: The answer contains significant fabrication or contradicts the context.

You MUST respond with ONLY a JSON object in this exact format:
{"score": <int 1-5>, "reasoning": "<brief explanation>"}\
"""


def render(query: str, answer: str, context_chunks: list[str]) -> tuple[str, str]:
    """Render the faithfulness judge prompt.

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

Evaluate the faithfulness of the answer. Respond with JSON only."""

    return SYSTEM_PROMPT, user_prompt
