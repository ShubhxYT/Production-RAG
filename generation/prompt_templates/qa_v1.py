"""Standard Q&A prompt template v1."""

from generation.models import PromptVariant, RenderedPrompt
from retrieval.models import RetrievalResult

VERSION = "qa_v1"

SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions based exclusively on \
the provided context documents.

Rules:
1. Answer ONLY based on the information in the provided context. Do not use \
any prior knowledge or make assumptions beyond what is stated.
2. If the context does not contain enough information to fully answer the \
question, clearly state what information is available and what is missing.
3. Cite your sources using the format [Source: Document Title, Page X] after \
each claim or piece of information.
4. Be concise and direct. Avoid unnecessary filler.
5. At the end of your answer, list all sources used under a "Sources:" heading.\
"""

USER_TEMPLATE = """\
Context:
{context_block}

Question: {query}

Provide a detailed answer based on the context above, citing sources.\
"""


def render(
    query: str,
    context_chunks: list[RetrievalResult],
) -> RenderedPrompt:
    """Render the Q&A prompt with context chunks.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks to include as context.

    Returns:
        RenderedPrompt with system and user prompts.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        source_label = chunk.document_title or chunk.source_path
        page_info = ""
        if chunk.page_numbers:
            pages = ", ".join(str(p) for p in chunk.page_numbers)
            page_info = f", Page {pages}"
        header = f"[Source {i}: {source_label}{page_info}]"
        context_parts.append(f"{header}\n{chunk.text}")

    context_block = "\n\n---\n\n".join(context_parts)

    return RenderedPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_TEMPLATE.format(
            context_block=context_block,
            query=query,
        ),
        variant=PromptVariant.QA,
        version=VERSION,
    )
