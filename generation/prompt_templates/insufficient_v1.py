"""Insufficient context fallback prompt template v1."""

from generation.models import PromptVariant, RenderedPrompt
from retrieval.models import RetrievalResult

VERSION = "insufficient_v1"

SYSTEM_PROMPT = """\
You are a helpful assistant. The retrieval system did not find sufficiently \
relevant documents to answer the user's question with confidence.

Rules:
1. Clearly state that the available documents do not contain enough \
information to answer the question confidently.
2. If any partially relevant information was found, briefly mention what \
was found and why it may not fully answer the question.
3. Suggest how the user might rephrase the question or what additional \
documents might help.
4. Do NOT fabricate or guess an answer.\
"""

USER_TEMPLATE = """\
Context (low confidence matches):
{context_block}

Question: {query}

The retrieved documents had low relevance scores. Respond accordingly.\
"""


def render(
    query: str,
    context_chunks: list[RetrievalResult],
) -> RenderedPrompt:
    """Render the insufficient-context fallback prompt.

    Args:
        query: The user's question.
        context_chunks: Low-confidence retrieval results.

    Returns:
        RenderedPrompt with system and user prompts.
    """
    if context_chunks:
        context_parts: list[str] = []
        for i, chunk in enumerate(context_chunks, 1):
            source_label = chunk.document_title or chunk.source_path
            header = f"[Source {i}: {source_label}]"
            context_parts.append(f"{header}\n{chunk.text[:200]}...")
        context_block = "\n\n---\n\n".join(context_parts)
    else:
        context_block = "(No relevant documents found.)"

    return RenderedPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_TEMPLATE.format(
            context_block=context_block,
            query=query,
        ),
        variant=PromptVariant.INSUFFICIENT,
        version=VERSION,
    )
