"""Transcript fallback prompt template v1."""

from generation.models import PromptVariant, RenderedPrompt
from retrieval.models import RetrievalResult

VERSION = "transcript_fallback_v1"

SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions based exclusively on \
the provided transcript context.

Rules:
1. Answer ONLY based on the information in the provided transcript. Do not use \
any prior knowledge or make assumptions beyond what is stated.
2. If the transcript does not contain enough information to fully answer the \
question, clearly state what information is available and what is missing.
3. Cite your sources using the format [Source: transcript] after each claim \
or piece of information.
4. Be concise and direct. Avoid unnecessary filler.
5. At the end of your answer, list all sources under a "Sources:" heading \
as: [Source: transcript].\
"""

USER_TEMPLATE = """\
Context:
{context_block}

Question: {query}

Provide a detailed answer based on the transcript context above, citing the source.\
"""


def render(
    query: str,
    context_chunks: list[RetrievalResult],
) -> RenderedPrompt:
    """Render the transcript fallback prompt with synthetic transcript chunks.

    Args:
        query: The user's question.
        context_chunks: Synthetic RetrievalResult chunks loaded from transcript.md.

    Returns:
        RenderedPrompt with system and user prompts.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        header = f"[Source {i}: transcript]"
        context_parts.append(f"{header}\n{chunk.text}")

    context_block = "\n\n---\n\n".join(context_parts)

    return RenderedPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_TEMPLATE.format(
            context_block=context_block,
            query=query,
        ),
        variant=PromptVariant.TRANSCRIPT_FALLBACK,
        version=VERSION,
    )
