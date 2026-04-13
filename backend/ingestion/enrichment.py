"""LLM-driven metadata enrichment for document chunks."""

import logging
import time

from generation.llm_service import EnrichmentProvider, GeminiProvider
from generation.models import LLMConfig
from generation.prompts import ENRICHMENT_SYSTEM_PROMPT
from ingestion.models import Chunk, Document

logger = logging.getLogger(__name__)


def enrich_chunks(
    chunks: list[Chunk],
    llm_service: EnrichmentProvider,
    batch_delay: float = 0.5,
) -> list[Chunk]:
    """Enrich chunks with LLM-generated metadata.

    For each chunk, generates a summary, keywords, and hypothetical
    questions using the LLM. Skips chunks that already have a
    non-empty summary. On per-chunk failure (after retries inside
    the provider), logs a warning and continues.

    Args:
        chunks: List of Chunk objects to enrich.
        llm_service: EnrichmentProvider instance for LLM calls.
        batch_delay: Seconds to wait between API calls (rate limiting).

    Returns:
        The same list of chunks with enrichment fields populated.
    """
    total = len(chunks)
    enriched = 0
    skipped = 0
    failed = 0

    for i, chunk in enumerate(chunks, start=1):
        # Skip already-enriched chunks
        if chunk.summary:
            skipped += 1
            logger.debug("Skipping chunk %d/%d (already enriched)", i, total)
            continue

        try:
            logger.info("Enriching chunk %d/%d ...", i, total)
            result = llm_service.enrich_chunk(
                chunk.text, ENRICHMENT_SYSTEM_PROMPT
            )
            chunk.summary = result.summary
            chunk.keywords = result.keywords
            chunk.hypothetical_questions = result.hypothetical_questions
            enriched += 1
        except Exception:
            failed += 1
            logger.warning(
                "Failed to enrich chunk %d/%d (id=%s), skipping",
                i,
                total,
                chunk.id,
                exc_info=True,
            )

        # Rate-limit delay (skip after the last chunk)
        if i < total and batch_delay > 0:
            time.sleep(batch_delay)

    logger.info(
        "Enrichment complete: %d enriched, %d skipped, %d failed (of %d total)",
        enriched,
        skipped,
        failed,
        total,
    )
    return chunks


def enrich_document(
    document: Document,
    llm_service: EnrichmentProvider | None = None,
    config: LLMConfig | None = None,
    batch_delay: float = 0.5,
) -> Document:
    """Enrich all chunks in a document with LLM-generated metadata.

    Args:
        document: Document with populated chunks.
        llm_service: EnrichmentProvider instance. Created if None.
        config: LLM configuration. Uses defaults if None.
        batch_delay: Seconds between API calls.

    Returns:
        The document with enriched chunks.
    """
    if not document.chunks:
        logger.info("No chunks to enrich in document %s", document.id)
        return document

    if llm_service is None:
        from generation.llm_service import get_enrichment_provider
        llm_service = get_enrichment_provider(config=config)

    enrich_chunks(document.chunks, llm_service, batch_delay=batch_delay)
    return document
