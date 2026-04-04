# Metadata Enrichment

## Goal
Add LLM-driven metadata enrichment to chunks — summaries, keywords, and hypothetical questions — using Google Gemini Flash with structured JSON output, enabling question-to-question matching during retrieval.

## Prerequisites
Make sure that the user is currently on the `feat/metadata-enrichment` branch before beginning implementation.
If not, move them to the correct branch. If the branch does not exist, create it from main.

## Implementation Steps

### Step 1: Add Dependencies and Environment Configuration

- [x] Open `pyproject.toml` and replace the `dependencies` list with:

```toml
dependencies = [
    "beautifulsoup4>=4.12.0",
    "google-genai>=1.70.0",
    "langchain-docling>=2.0.0",
    "langchain-opendataloader-pdf>=2.0.0",
    "liteparse>=1.2.1",
    "markdown-it-py>=3.0.0",
    "marker-pdf>=1.10.2",
    "markdownify>=0.13.0",
    "pydantic>=2.0.0",
    "python-docx>=1.0.0",
    "python-dotenv>=1.0.0",
    "sentence-transformers>=3.0.0",
    "tiktoken>=0.7.0",
    "torch>=2.11.0",
    "torchvision>=0.26.0",
]
```

- [x] Run dependency sync:

```bash
uv sync
```

- [x] Create file `.env.example`:

```env
# Google Gemini — used for chunk metadata enrichment (summaries, keywords, questions)
GEMINI_API_KEY=your-gemini-api-key-here

# Cerebras Cloud — reserved for answer synthesis (Step 9+)
CEREBRAS_API_KEY=your-cerebras-api-key-here
```

- [x] Create your local `.env` file with real keys (this file is gitignored):

```bash
cp .env.example .env
# Edit .env and paste your real API keys
```

- [x] Add `.env` to `.gitignore` if not already present:

```bash
echo ".env" >> .gitignore
```

- [x] Create file `config/__init__.py`:

```python
"""Configuration management for the FullRag system."""
```

- [x] Create file `config/settings.py`:

```python
"""Environment-based settings loader."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def get_gemini_api_key() -> str:
    """Return the Gemini API key from environment.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set.
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key


def get_cerebras_api_key() -> str:
    """Return the Cerebras API key from environment.

    Raises:
        RuntimeError: If CEREBRAS_API_KEY is not set.
    """
    key = os.environ.get("CEREBRAS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "CEREBRAS_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key
```

##### Step 1 Verification Checklist
- [x] `uv sync` completes without errors
- [x] `python -c "from google import genai; print('OK')"` prints `OK`
- [x] `python -c "from dotenv import load_dotenv; print('OK')"` prints `OK`
- [x] `python -c "from config.settings import get_gemini_api_key; print(get_gemini_api_key()[:8] + '...')"` prints your key prefix (with `.env` populated)

#### Step 1 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `feat: add google-genai, python-dotenv deps and config settings`

Update README.md iteration log:
```markdown
### Metadata Enrichment — Step 1: Dependencies & Config

- Added `google-genai>=1.70.0` and `python-dotenv>=1.0.0` to `pyproject.toml`
- Created `.env.example` with `GEMINI_API_KEY` and `CEREBRAS_API_KEY` placeholders
- Created `config/settings.py` for environment-based API key loading
```

---

### Step 2: Update Chunk Model with Enrichment Fields

- [x] Open `ingestion/models.py` and replace the entire `Chunk` class with:

```python
class Chunk(BaseModel):
    """A chunk of text produced by the structure-aware chunker."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    token_count: int
    document_id: str
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    element_types: list[ElementType] = Field(default_factory=list)
    position: int = 0
    overlap_before: str = ""

    # Enrichment fields (populated by LLM in Step 5)
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    hypothetical_questions: list[str] = Field(default_factory=list)
```

- [x] In the same file, replace the `Document` class with:

```python
class Document(BaseModel):
    """A parsed document with structural elements and metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    title: str | None = None
    format: str
    elements: list[Element] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    raw_content: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Document-level metadata (for future filtering)
    doc_date: str | None = None
    doc_version: str | None = None
```

##### Step 2 Verification Checklist
- [x] Existing staging JSONs still load: `python -c "from ingestion.staging import load_staged_document; from pathlib import Path; d = load_staged_document(list(Path('staging').glob('*.json'))[0]); print(f'Loaded {d.title}, chunks={len(d.chunks)}')"` — prints doc title and chunk count
- [x] New fields default correctly: `python -c "from ingestion.models import Chunk; c = Chunk(text='test', token_count=5, document_id='x'); print(c.summary, c.keywords, c.hypothetical_questions)"` — prints empty string, empty list, empty list
- [x] Existing tests pass: `python -m pytest test/test_chunker.py -q`

#### Step 2 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `feat: add enrichment fields (summary, keywords, questions) to Chunk model`

Update README.md iteration log:
```markdown
### Metadata Enrichment — Step 2: Updated Chunk Model

- Added `summary`, `keywords`, `hypothetical_questions` fields to `Chunk` model (default to empty)
- Added `doc_date`, `doc_version` fields to `Document` model
- Backward compatible: existing staged JSONs without these fields still load correctly
```

---

### Step 3: Create Generation Module with Gemini LLM Service

- [x] Create file `generation/__init__.py`:

```python
"""LLM generation service for the FullRag system."""
```

- [x] Create file `generation/models.py`:

```python
"""Data models for the generation service."""

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM-based enrichment."""

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 1024


class ChunkEnrichment(BaseModel):
    """Structured enrichment output from the LLM.

    Used as Gemini's response_json_schema to guarantee
    structured output without manual JSON parsing.
    """

    summary: str = Field(
        description="A concise 1-2 sentence summary of what the chunk contains."
    )
    keywords: list[str] = Field(
        description="3-7 specific, domain-relevant keywords extracted from the chunk."
    )
    hypothetical_questions: list[str] = Field(
        description="2-5 hypothetical questions that this chunk could answer."
    )
```

- [x] Create file `generation/llm_service.py`:

```python
"""LLM service with Gemini provider for chunk enrichment."""

import json
import logging
import time

from google import genai
from google.genai import types

from config.settings import get_gemini_api_key
from generation.models import ChunkEnrichment, LLMConfig

logger = logging.getLogger(__name__)


class GeminiProvider:
    """Wraps the Google GenAI client for structured chunk enrichment.

    Uses Gemini's native JSON schema output mode to guarantee
    structured responses conforming to ChunkEnrichment.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = genai.Client(api_key=get_gemini_api_key())

    def enrich_chunk(
        self, chunk_text: str, system_prompt: str
    ) -> ChunkEnrichment:
        """Generate enrichment metadata for a single chunk.

        Args:
            chunk_text: The chunk text to enrich.
            system_prompt: System instruction for the LLM.

        Returns:
            ChunkEnrichment with summary, keywords, and questions.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=chunk_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                        response_mime_type="application/json",
                        response_json_schema=ChunkEnrichment.model_json_schema(),
                    ),
                )

                # Log token usage
                if response.usage_metadata:
                    logger.debug(
                        "Token usage: prompt=%d, completion=%d, total=%d",
                        response.usage_metadata.prompt_token_count or 0,
                        response.usage_metadata.candidates_token_count or 0,
                        response.usage_metadata.total_token_count or 0,
                    )

                # Parse structured JSON response
                raw_text = response.text
                if not raw_text:
                    raise ValueError("Empty response from Gemini")

                data = json.loads(raw_text)
                return ChunkEnrichment.model_validate(data)

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt,
                        max_retries,
                        e,
                        wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Enrichment failed after {max_retries} retries: {last_error}"
        )
```

##### Step 3 Verification Checklist
- [x] `python -c "from generation.models import LLMConfig, ChunkEnrichment; print('OK')"` prints `OK`
- [x] `python -c "from generation.models import ChunkEnrichment; print(ChunkEnrichment.model_json_schema())"` prints a valid JSON schema
- [x] `python -c "from generation.llm_service import GeminiProvider; print('OK')"` prints `OK` (requires `.env` with `GEMINI_API_KEY`)

#### Step 3 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `feat: create generation module with Gemini LLM service and structured output`

Update README.md iteration log:
```markdown
### Metadata Enrichment — Step 3: Generation Module

- Created `generation/` module with `LLMConfig` and `ChunkEnrichment` Pydantic models
- Created `GeminiProvider` wrapping `google.genai.Client` for structured JSON enrichment
- Uses Gemini's native `response_json_schema` — no manual JSON parsing needed
- 3-retry exponential backoff for transient API errors
- Token usage logged at DEBUG level
```

---

### Step 4: Define Enrichment Prompt

- [x] Create file `generation/prompts.py`:

```python
"""Prompt templates for LLM-driven enrichment."""

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
```

##### Step 4 Verification Checklist
- [x] `python -c "from generation.prompts import ENRICHMENT_SYSTEM_PROMPT; print(len(ENRICHMENT_SYSTEM_PROMPT), 'chars')"` — prints a non-zero character count
- [x] Quick quality test with a real API call:

```bash
python -c "
from generation.llm_service import GeminiProvider
from generation.prompts import ENRICHMENT_SYSTEM_PROMPT

provider = GeminiProvider()
result = provider.enrich_chunk(
    'RAG or retrieval augmented generation solves this. When a user asks a question, you first retrieve relevant pieces of information from your own documents. Then you augment the question with that retrieved context. And finally, you let the LLM generate an answer based on what you gave it.',
    ENRICHMENT_SYSTEM_PROMPT,
)
print('Summary:', result.summary)
print('Keywords:', result.keywords)
print('Questions:', result.hypothetical_questions)
"
```

- [x] Verify: summary reflects chunk content, keywords are specific, questions are answerable by the chunk

#### Step 4 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `feat: add enrichment system prompt for metadata generation`

Update README.md iteration log:
```markdown
### Metadata Enrichment — Step 4: Enrichment Prompt

- Created `generation/prompts.py` with `ENRICHMENT_SYSTEM_PROMPT`
- Instructs LLM to generate summary (1-2 sentences), keywords (3-7), hypothetical questions (2-5)
- Emphasizes no hallucination, specific keywords, natural user questions
```

---

### Step 5: Implement Enrichment Pipeline

- [x] Create file `ingestion/enrichment.py`:

```python
"""LLM-driven metadata enrichment for document chunks."""

import logging
import time

from generation.llm_service import GeminiProvider
from generation.models import LLMConfig
from generation.prompts import ENRICHMENT_SYSTEM_PROMPT
from ingestion.models import Chunk, Document

logger = logging.getLogger(__name__)


def enrich_chunks(
    chunks: list[Chunk],
    llm_service: GeminiProvider,
    batch_delay: float = 0.5,
) -> list[Chunk]:
    """Enrich chunks with LLM-generated metadata.

    For each chunk, generates a summary, keywords, and hypothetical
    questions using the LLM. Skips chunks that already have a
    non-empty summary. On per-chunk failure (after retries inside
    the provider), logs a warning and continues.

    Args:
        chunks: List of Chunk objects to enrich.
        llm_service: GeminiProvider instance for LLM calls.
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
    llm_service: GeminiProvider | None = None,
    config: LLMConfig | None = None,
    batch_delay: float = 0.5,
) -> Document:
    """Enrich all chunks in a document with LLM-generated metadata.

    Args:
        document: Document with populated chunks.
        llm_service: GeminiProvider instance. Created if None.
        config: LLM configuration. Uses defaults if None.
        batch_delay: Seconds between API calls.

    Returns:
        The document with enriched chunks.
    """
    if not document.chunks:
        logger.info("No chunks to enrich in document %s", document.id)
        return document

    if llm_service is None:
        llm_service = GeminiProvider(config=config)

    enrich_chunks(document.chunks, llm_service, batch_delay=batch_delay)
    return document
```

##### Step 5 Verification Checklist
- [x] `python -c "from ingestion.enrichment import enrich_chunks, enrich_document; print('OK')"` prints `OK`
- [x] Quick integration test — enrich chunks from a staged document:

```bash
python -c "
from ingestion.staging import load_staged_document
from ingestion.enrichment import enrich_document
from pathlib import Path

doc = load_staged_document(list(Path('staging').glob('*.json'))[0])
print(f'Document: {doc.title}, chunks: {len(doc.chunks)}')
doc = enrich_document(doc, batch_delay=1.0)
for c in doc.chunks[:2]:
    print(f'\n--- Chunk {c.position} ---')
    print(f'Summary: {c.summary}')
    print(f'Keywords: {c.keywords}')
    print(f'Questions: {c.hypothetical_questions}')
"
```

- [x] Verify summaries accurately reflect chunk content and questions are answerable

#### Step 5 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `feat: implement enrichment pipeline with skip logic and graceful failure`

Update README.md iteration log:
```markdown
### Metadata Enrichment — Step 5: Enrichment Pipeline

- Created `ingestion/enrichment.py` with `enrich_chunks()` and `enrich_document()`
- Sequential processing with configurable rate-limit delay between API calls
- Skips already-enriched chunks (idempotent re-runs)
- Per-chunk failure is non-fatal: logs warning, continues batch
```

---

### Step 6: Add CLI Enrich Subcommand

- [x] Replace the entire contents of `ingestion/cli.py` with:

```python
"""CLI for the ingestion pipeline."""

import argparse
import logging
import sys
import time
from pathlib import Path

from ingestion.pipeline import IngestionPipeline


def _run_ingest(args: argparse.Namespace) -> None:
    """Run the ingest subcommand."""
    input_path = Path(args.input)
    pipeline = IngestionPipeline(output_dir=args.output)

    if input_path.is_dir():
        documents = pipeline.ingest_directory(
            input_path, skip_existing=args.skip_existing
        )
    elif input_path.is_file():
        doc = pipeline.ingest_file(
            input_path, skip_existing=args.skip_existing
        )
        documents = [doc] if doc else []
    else:
        print(f"Error: path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Print element summary
    total_elements = sum(len(d.elements) for d in documents)
    element_counts: dict[str, int] = {}
    for doc in documents:
        for el in doc.elements:
            key = el.type.value
            element_counts[key] = element_counts.get(key, 0) + 1

    print(f"\n{'=' * 50}")
    print("Ingestion Summary")
    print(f"{'=' * 50}")
    print(f"Files processed: {len(documents)}")
    print(f"Total elements:  {total_elements}")
    if element_counts:
        print("\nElement breakdown:")
        for etype, count in sorted(element_counts.items()):
            print(f"  {etype:15s} {count}")

    # Print chunk summary
    total_chunks = sum(len(d.chunks) for d in documents)
    if total_chunks > 0:
        all_token_counts = [
            c.token_count for d in documents for c in d.chunks
        ]
        min_tokens = min(all_token_counts)
        max_tokens = max(all_token_counts)
        avg_tokens = sum(all_token_counts) / len(all_token_counts)

        chunk_element_counts: dict[str, int] = {}
        for doc in documents:
            for chunk in doc.chunks:
                for et in chunk.element_types:
                    key = et.value
                    chunk_element_counts[key] = (
                        chunk_element_counts.get(key, 0) + 1
                    )

        print(f"\n{'─' * 50}")
        print("Chunking Summary")
        print(f"{'─' * 50}")
        print(f"Total chunks:    {total_chunks}")
        print(f"Token counts:    min={min_tokens}  max={max_tokens}  avg={avg_tokens:.0f}")
        if chunk_element_counts:
            print("\nElement types in chunks:")
            for etype, count in sorted(chunk_element_counts.items()):
                print(f"  {etype:15s} {count}")

    print(f"{'=' * 50}")


def _run_enrich(args: argparse.Namespace) -> None:
    """Run the enrich subcommand."""
    from generation.llm_service import GeminiProvider
    from generation.models import LLMConfig
    from ingestion.enrichment import enrich_chunks
    from ingestion.staging import load_staged_document, stage_document

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.is_dir():
        print(
            f"Error: staging directory does not exist: {args.input}",
            file=sys.stderr,
        )
        sys.exit(1)

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load staged documents
    documents = []
    for jf in json_files:
        try:
            doc = load_staged_document(jf)
            documents.append(doc)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to load: %s", jf.name
            )

    # Collect all chunks across documents
    all_chunks = [c for doc in documents for c in doc.chunks]
    if not all_chunks:
        print("No chunks found in staged documents.", file=sys.stderr)
        sys.exit(1)

    # Set up LLM service
    config = LLMConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )
    provider = GeminiProvider(config=config)

    # Enrich all chunks
    start_time = time.perf_counter()
    enrich_chunks(all_chunks, provider, batch_delay=args.delay)
    elapsed = time.perf_counter() - start_time

    # Count results
    enriched = sum(1 for c in all_chunks if c.summary)
    total = len(all_chunks)

    # Re-save documents with enriched chunks
    saved = 0
    for doc in documents:
        stage_document(doc, staging_dir=str(output_path))
        saved += 1

    # Print summary
    print(f"\n{'=' * 50}")
    print("Enrichment Summary")
    print(f"{'=' * 50}")
    print(f"Documents:         {len(documents)}")
    print(f"Total chunks:      {total}")
    print(f"Enriched:          {enriched}")
    print(f"Skipped/Failed:    {total - enriched}")
    print(f"Model:             {config.model_name}")
    print(f"Elapsed time:      {elapsed:.1f}s")
    print(f"Output directory:  {output_path}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ingestion",
        description="FullRag ingestion pipeline: ingest documents and enrich chunks.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- ingest subcommand --
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents from a file or directory."
    )
    ingest_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a file or directory to ingest.",
    )
    ingest_parser.add_argument(
        "--output",
        "-o",
        default="results",
        help="Output directory for processed documents (default: results).",
    )
    ingest_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that have already been processed.",
    )

    # -- enrich subcommand --
    enrich_parser = subparsers.add_parser(
        "enrich", help="Enrich staged document chunks with LLM-generated metadata."
    )
    enrich_parser.add_argument(
        "--input",
        "-i",
        default="staging",
        help="Path to staging directory with Document JSON files (default: staging).",
    )
    enrich_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for enriched JSONs (default: same as input).",
    )
    enrich_parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash).",
    )
    enrich_parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3).",
    )
    enrich_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max output tokens (default: 1024).",
    )
    enrich_parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between API calls for rate limiting (default: 0.5).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "enrich":
        _run_enrich(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [x] Update `ingestion/__main__.py` (no changes needed — it already calls `main()`):

```python
"""Allow running the ingestion module with python -m ingestion."""

from ingestion.cli import main

main()
```

##### Step 6 Verification Checklist
- [x] `python -m ingestion --help` shows both `ingest` and `enrich` subcommands
- [x] `python -m ingestion enrich --help` shows `--input`, `--output`, `--model`, `--delay` options
- [x] Run enrichment on staging documents:

```bash
python -m ingestion enrich --input staging/ --delay 1.0
```

- [x] Verify enriched JSON: `python -c "from ingestion.staging import load_staged_document; from pathlib import Path; d = load_staged_document(list(Path('staging').glob('*.json'))[0]); c = d.chunks[0]; print('Summary:', c.summary); print('Keywords:', c.keywords); print('Questions:', c.hypothetical_questions)"`
- [x] Run enrichment again and verify all chunks are skipped (already enriched)
- [x] Verify old `ingest` still works: `python -m ingestion ingest --help`

#### Step 6 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `feat: add CLI enrich subcommand for batch metadata enrichment`

Update README.md iteration log:
```markdown
### Metadata Enrichment — Step 6: CLI Enrich Subcommand

- Refactored `ingestion/cli.py` to use subcommands: `ingest` and `enrich`
- `python -m ingestion enrich --input staging/` enriches all staged document chunks
- Supports `--model`, `--delay`, `--temperature`, `--max-tokens` flags
- Re-saves enriched documents to staging directory (or custom output)
- Prints summary stats: enriched, skipped, failed, elapsed time
```

---

### Step 7: Add Unit Tests

- [x] Create file `test/test_enrichment.py`:

```python
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
```

##### Step 7 Verification Checklist
- [x] Run enrichment tests:

```bash
python -m pytest test/test_enrichment.py -v
```

- [x] All tests pass
- [x] Existing tests still pass:

```bash
python -m pytest test/test_chunker.py test/test_embeddings.py -v
```

#### Step 7 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.

Suggested commit: `feat: add unit tests for metadata enrichment with mock provider`

Update README.md iteration log:
```markdown
### Metadata Enrichment — Step 7: Unit Tests

- Created `test/test_enrichment.py` with 14 tests covering:
  - `ChunkEnrichment` model validation and roundtrip serialization
  - `LLMConfig` default values
  - Mock provider for testing without API calls
  - Single/multiple chunk enrichment
  - Skip-already-enriched logic
  - Continue-on-failure behavior
  - Backward compatibility (old JSONs without enrichment fields)
  - Prompt constant validation
- All existing tests (`test_chunker.py`, `test_embeddings.py`) remain passing
```

---

## Final Verification

After all steps are committed:

1. `uv sync` — clean install
2. `python -m pytest test/ -v` — all tests pass
3. `python -m ingestion ingest -i data/transcript.md` — ingestion still works
4. `python -m ingestion enrich --input staging/` — enriches all chunks
5. Inspect any staged JSON — verify `summary`, `keywords`, `hypothetical_questions` are populated
6. Re-run enrich — verify all chunks skipped (idempotent)
