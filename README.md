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
