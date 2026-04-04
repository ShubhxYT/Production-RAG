# FullRag

## Copilot Iteration Workflow

This repository includes a GitHub Copilot skill that enforces an iterative workflow:

1. Complete one meaningful iteration.
2. Update `README.md` with what changed.
3. Commit that iteration before starting the next one.

Skill location:

- `.github/skills/commit-after-each-iteration/SKILL.md`

Expected commit message format:

- `iter-N: <short summary of change>`

Example:

- `iter-1: add PDF parsing pipeline scaffold`

## Iteration Log

### Metadata Enrichment - Step 7: Unit Tests

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

### Metadata Enrichment - Step 6: CLI Enrich Subcommand

- Refactored `ingestion/cli.py` to use subcommands: `ingest` and `enrich`
- `python -m ingestion enrich --input staging/` enriches all staged document chunks
- Supports `--model`, `--delay`, `--temperature`, `--max-tokens` flags
- Re-saves enriched documents to staging directory (or custom output)
- Prints summary stats: enriched, skipped, failed, elapsed time

### Metadata Enrichment - Step 5: Enrichment Pipeline

- Created `ingestion/enrichment.py` with `enrich_chunks()` and `enrich_document()`
- Sequential processing with configurable rate-limit delay between API calls
- Skips already-enriched chunks (idempotent re-runs)
- Per-chunk failure is non-fatal: logs warning, continues batch

### Metadata Enrichment - Step 4: Enrichment Prompt

- Created `generation/prompts.py` with `ENRICHMENT_SYSTEM_PROMPT`
- Instructs LLM to generate summary (1-2 sentences), keywords (3-7), hypothetical questions (2-5)
- Emphasizes no hallucination, specific keywords, natural user questions

### Metadata Enrichment - Step 3: Generation Module

- Created `generation/` module with `LLMConfig` and `ChunkEnrichment` Pydantic models
- Created `GeminiProvider` wrapping `google.genai.Client` for structured JSON enrichment
- Uses Gemini's native `response_json_schema` - no manual JSON parsing needed
- 3-retry exponential backoff for transient API errors
- Token usage logged at DEBUG level

### Metadata Enrichment - Step 2: Updated Chunk Model

- Added `summary`, `keywords`, `hypothetical_questions` fields to `Chunk` model (default to empty)
- Added `doc_date`, `doc_version` fields to `Document` model
- Backward compatible: existing staged JSONs without these fields still load correctly

### Metadata Enrichment - Step 1: Dependencies & Config

- Added `google-genai>=1.70.0` and `python-dotenv>=1.0.0` to `pyproject.toml`
- Created `.env.example` with `GEMINI_API_KEY` and `CEREBRAS_API_KEY` placeholders
- Created `config/settings.py` for environment-based API key loading

### iter-13: Gitignore Updates

- Added `.embedding_cache/` and `.embedding_output/` to `.gitignore`

### iter-12: Embedding Unit Tests

- Created `test/test_embeddings.py` with tests for:
	- Model defaults (`BAAI/bge-base-en-v1.5`, 768 dims, zero token cost)
	- Batching logic (empty, single, multiple, exact boundary, 250->3 batches)
	- Cache key determinism and uniqueness
	- Cache hit/miss/partial-hit behavior
	- Cache persistence to disk across instances
- All tests use a `MockProvider` - no GPU or model download required

**How to run:**

```bash
pytest test/test_embeddings.py -v
```

### iter-11: Embedding CLI

- Created `embeddings/cli.py` and `embeddings/__main__.py`
- CLI command: `python -m embeddings -i staging/ -o .embedding_output`
- Loads staged Document JSON files, extracts chunks, embeds via local SentenceTransformer on CUDA
- Supports `--no-cache`, `--batch-size`, `--model` options
- Outputs `embeddings.json` mapping chunk IDs to vectors (plus device metadata)
- Prints summary: document count, chunk count, device, elapsed time, $0.00 API cost
- No API key required

**How to run:**

```bash
# Embed all staged chunks (uses CUDA automatically)
python -m embeddings -i staging/ -o .embedding_output

# With verbose logging and no cache
python -m embeddings -i staging/ -o .embedding_output --no-cache -v

# Use a smaller/faster model
python -m embeddings -i staging/ -o .embedding_output --model BAAI/bge-small-en-v1.5
```

### iter-10: Embedding Cache

- Created `embeddings/cache.py` with `CachedEmbeddingService`
- File-based JSON cache keyed by SHA-256 of (model_name, text)
- Separates cached/uncached texts; only sends uncached to provider
- Cache stored at `.embedding_cache/cache.json` by default

### iter-9: Embedding Service Implementation

- Created `embeddings/service.py` with:
	- `EmbeddingProvider` protocol for provider abstraction
	- `SentenceTransformerProvider` using `sentence-transformers` on local CUDA GPU (GTX 1650 SUPER; falls back to CPU)
	- `EmbeddingService` with automatic batching (configurable batch size)
	- Model is loaded lazily and cached in memory; CUDA device auto-detected via `torch.cuda.is_available()`
	- No API key required

### iter-8: Embeddings Module Models

- Created `embeddings/` module with `EmbeddingConfig` and `EmbeddingResult` Pydantic models
- `EmbeddingConfig`: model_name (`BAAI/bge-base-en-v1.5`), dimensions (768), batch_size, max_retries, cache_dir
- `EmbeddingResult`: vectors, model, dimensions, token_usage

### iter-7: Embedding Service Dependencies

- Added `sentence-transformers>=3.0.0` to `pyproject.toml`
- Enables local GPU (CUDA) embedding generation - no API key required
- `torch` was already a dependency; CUDA device is used automatically when available (GTX 1650 SUPER)

### iter-2

- Updated `.gitignore` so `data/` and `results/` stay in the repository while their contents are ignored.
- Added placeholder files: `data/.gitkeep` and `results/.gitkeep`.
- Intended usage: keep directory structure in git without committing generated files or datasets inside these folders.

## Evaluation

### Retrieval Evaluation

Measures retrieval quality using standard information retrieval metrics against a ground-truth dataset.

**Metrics:**
- **Precision@k** - Fraction of top-k results that are relevant
- **Recall@k** - Fraction of all relevant documents found in top-k
- **MRR** (Mean Reciprocal Rank) - Reciprocal of the rank of the first relevant result
- **NDCG@k** (Normalized Discounted Cumulative Gain) - Accounts for both relevance and ranking position

**Running:**
```bash
# Full evaluation with default settings
python -m evaluation

# With verbose per-query output
python -m evaluation --verbose

# Custom parameters
python -m evaluation --top-k 10 --k-values 1 3 5 10 --verbose

# Skip saving (dry run)
python -m evaluation --no-save --verbose
```

**Adding queries to the ground-truth dataset:**

Edit `evaluation/datasets/retrieval_ground_truth.json` and add new entries to the `annotations` array:
```json
{
	"query": "Your evaluation query here",
	"relevant_chunk_ids": ["chunk-uuid-1", "chunk-uuid-2"],
	"tags": ["factual"],
	"notes": "Why these chunks are relevant."
}
```

Use the chunk inspection script to find chunk IDs:
```bash
python -c "
from database.connection import get_session
from database.models import ChunkModel
from sqlalchemy import select
session = get_session()
for c in session.execute(select(ChunkModel).limit(20)).scalars():
		print(f'{c.id}: {c.text[:80]}...')
session.close()
"
```

**Reports** are saved to `evaluation/results/` as timestamped JSON files.
