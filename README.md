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
