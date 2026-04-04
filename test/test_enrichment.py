"""Unit tests for metadata enrichment."""

import json

from generation.models import ChunkEnrichment, LLMConfig
from ingestion.enrichment import enrich_chunks
from ingestion.models import Chunk, Document, ElementType


# ---------------------------------------------------------------------------
# ChunkEnrichment model
# ---------------------------------------------------------------------------


def test_chunk_enrichment_valid():
    enrichment = ChunkEnrichment(
        summary="This chunk discusses RAG systems.",
        keywords=["RAG", "retrieval", "augmentation"],
        hypothetical_questions=["What is RAG?"],
    )
    assert enrichment.summary == "This chunk discusses RAG systems."
    assert len(enrichment.keywords) == 3
    assert len(enrichment.hypothetical_questions) == 1


def test_chunk_enrichment_roundtrip():
    enrichment = ChunkEnrichment(
        summary="Summary text.",
        keywords=["a", "b", "c"],
        hypothetical_questions=["Q1?", "Q2?"],
    )
    data = enrichment.model_dump()
    restored = ChunkEnrichment.model_validate(data)
    assert restored.summary == enrichment.summary
    assert restored.keywords == enrichment.keywords
    assert restored.hypothetical_questions == enrichment.hypothetical_questions


def test_chunk_enrichment_json_schema():
    schema = ChunkEnrichment.model_json_schema()
    assert "summary" in schema["properties"]
    assert "keywords" in schema["properties"]
    assert "hypothetical_questions" in schema["properties"]


# ---------------------------------------------------------------------------
# LLMConfig defaults
# ---------------------------------------------------------------------------


def test_llm_config_defaults():
    config = LLMConfig()
    assert config.model_name == "gemini-2.5-flash"
    assert config.temperature == 0.3
    assert config.max_output_tokens == 1024


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockGeminiProvider:
    """Mock provider returning canned enrichment results."""

    def __init__(
        self,
        enrichment: ChunkEnrichment | None = None,
        fail_on: set[int] | None = None,
    ) -> None:
        self.call_count = 0
        self.last_texts: list[str] = []
        self._enrichment = enrichment or ChunkEnrichment(
            summary="Mock summary.",
            keywords=["mock", "test"],
            hypothetical_questions=["What is this?"],
        )
        self._fail_on = fail_on or set()

    def enrich_chunk(self, chunk_text: str, system_prompt: str) -> ChunkEnrichment:
        self.call_count += 1
        self.last_texts.append(chunk_text)
        if self.call_count in self._fail_on:
            raise RuntimeError(f"Simulated failure on call {self.call_count}")
        return self._enrichment


# ---------------------------------------------------------------------------
# enrich_chunks
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "Test chunk.", position: int = 0, **kwargs) -> Chunk:
    return Chunk(
        text=text,
        token_count=10,
        document_id="doc-1",
        position=position,
        **kwargs,
    )


def test_enrich_single_chunk():
    chunk = _make_chunk("Hello world.")
    provider = MockGeminiProvider()
    enrich_chunks([chunk], provider, batch_delay=0)

    assert chunk.summary == "Mock summary."
    assert chunk.keywords == ["mock", "test"]
    assert chunk.hypothetical_questions == ["What is this?"]
    assert provider.call_count == 1


def test_enrich_multiple_chunks():
    chunks = [_make_chunk(f"Chunk {i}.", position=i) for i in range(3)]
    provider = MockGeminiProvider()
    enrich_chunks(chunks, provider, batch_delay=0)

    assert all(c.summary == "Mock summary." for c in chunks)
    assert provider.call_count == 3


def test_enrich_skips_already_enriched():
    chunks = [
        _make_chunk("Already enriched.", summary="Existing summary."),
        _make_chunk("Needs enrichment."),
    ]
    provider = MockGeminiProvider()
    enrich_chunks(chunks, provider, batch_delay=0)

    assert chunks[0].summary == "Existing summary."  # unchanged
    assert chunks[1].summary == "Mock summary."  # enriched
    assert provider.call_count == 1  # only called for the second chunk


def test_enrich_continues_on_failure():
    chunks = [_make_chunk(f"Chunk {i}.", position=i) for i in range(3)]
    # Fail on the 2nd call (chunk index 1)
    provider = MockGeminiProvider(fail_on={2})
    enrich_chunks(chunks, provider, batch_delay=0)

    assert chunks[0].summary == "Mock summary."  # enriched
    assert chunks[1].summary == ""  # failed, stays empty
    assert chunks[2].summary == "Mock summary."  # enriched
    assert provider.call_count == 3


def test_enrich_empty_list():
    provider = MockGeminiProvider()
    result = enrich_chunks([], provider, batch_delay=0)
    assert result == []
    assert provider.call_count == 0


# ---------------------------------------------------------------------------
# Chunk model backward compatibility
# ---------------------------------------------------------------------------


def test_chunk_without_enrichment_fields():
    """Deserializing a JSON without enrichment fields should default to empty."""
    raw = json.dumps(
        {
            "id": "abc",
            "text": "Some chunk.",
            "token_count": 10,
            "document_id": "doc-1",
            "section_path": [],
            "page_numbers": [],
            "element_types": ["paragraph"],
            "position": 0,
            "overlap_before": "",
        }
    )
    chunk = Chunk.model_validate_json(raw)
    assert chunk.summary == ""
    assert chunk.keywords == []
    assert chunk.hypothetical_questions == []


def test_chunk_with_enrichment_fields():
    raw = json.dumps(
        {
            "id": "abc",
            "text": "Some chunk.",
            "token_count": 10,
            "document_id": "doc-1",
            "section_path": [],
            "page_numbers": [],
            "element_types": ["paragraph"],
            "position": 0,
            "overlap_before": "",
            "summary": "A summary.",
            "keywords": ["key1", "key2"],
            "hypothetical_questions": ["Q?"],
        }
    )
    chunk = Chunk.model_validate_json(raw)
    assert chunk.summary == "A summary."
    assert chunk.keywords == ["key1", "key2"]
    assert chunk.hypothetical_questions == ["Q?"]


# ---------------------------------------------------------------------------
# Document model backward compatibility
# ---------------------------------------------------------------------------


def test_document_without_doc_metadata():
    raw = json.dumps(
        {
            "id": "doc-1",
            "source_path": "test.md",
            "format": "md",
        }
    )
    doc = Document.model_validate_json(raw)
    assert doc.doc_date is None
    assert doc.doc_version is None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_enrichment_prompt_exists():
    from generation.prompts import ENRICHMENT_SYSTEM_PROMPT

    assert len(ENRICHMENT_SYSTEM_PROMPT) > 100
    assert "summary" in ENRICHMENT_SYSTEM_PROMPT.lower()
    assert "keywords" in ENRICHMENT_SYSTEM_PROMPT.lower()
    assert "question" in ENRICHMENT_SYSTEM_PROMPT.lower()
