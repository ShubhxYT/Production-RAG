# FullRag

FullRag is a modular Retrieval-Augmented Generation (RAG) pipeline focused on:

- multi-format ingestion
- structure-aware chunking
- optional LLM metadata enrichment
- local embedding generation (GPU/CPU)
- PostgreSQL + pgvector retrieval
- retrieval quality evaluation

This README is written bottom-up: internals first, then setup, then runbook.

## System Preview Image

![FullRag Architecture](fullrag-architecture.png)

## Project Structure (Bottom Layer)

- `ingestion/`: Loaders, restructuring, chunking, staging
- `generation/`: LLM config + enrichment prompts/provider
- `embeddings/`: Local sentence-transformer embedding service + cache + CLI
- `database/`: SQLAlchemy models, repository, seed command
- `retrieval/`: Query embedding + vector similarity search + REPL CLI
- `evaluation/`: Retrieval metrics and evaluation runner
- `alembic/`: DB migrations
- `test/`: Unit and integration tests
- `staging/`: Staged document JSON payloads
- `results/`: Output artifacts from ingestion and experiments

## Data Flow (Core)

1. Ingest files (`pdf`, `docx`, `html`, `htm`, `md`) into normalized document elements.
2. Chunk documents into retrievable text units with metadata.
3. Optionally enrich chunks using Gemini (summary, keywords, hypothetical questions).
4. Seed PostgreSQL with document records, chunks, and vector embeddings.
5. Retrieve with semantic similarity (`pgvector`) via CLI (single query or REPL).
6. Evaluate retrieval quality with Precision@k, Recall@k, MRR, NDCG@k.

## Database Layer

- Engine: PostgreSQL + `pgvector`
- Vector index: HNSW (`vector_cosine_ops`)
- Main tables:
  - `documents`
  - `chunks`
  - `chunk_embeddings`

Default local database URL:

`postgresql+psycopg://fullrag:fullrag@localhost:5432/fullrag`

## Environment Variables

Create `.env` from `.env.example` and set:

- `GEMINI_API_KEY` (required only for enrichment)
- `GROQ_API_KEY` (required for RAG answer generation — get one at console.groq.com)
- `DATABASE_URL` (optional if you use local default)

## Setup

### 1) Install dependencies

```bash
uv sync
```

If you do not use `uv`, install from `pyproject.toml` in your preferred workflow.

### 2) Start PostgreSQL with pgvector

```bash
docker compose -f pgvector.yaml up -d
```

### 3) Apply migrations

```bash
alembic upgrade head
```

### 4) Configure environment

```bash
cp .env.example .env
# edit .env and add keys as needed
```

## Runbook

### Ingest documents

```bash
python -m ingestion ingest -i data -o results
```

### Enrich staged chunks (optional)

```bash
python -m ingestion enrich -i staging
```

### Generate embeddings

```bash
python -m embeddings -i staging -o .embedding_output
```

### Seed database

```bash
python -m database seed --staging-dir staging
```

### Retrieve (single query)

```bash
python -m retrieval -q "your question here" -k 5
```

### Retrieve (interactive REPL)

```bash
python -m retrieval
```

### Evaluate retrieval

```bash
python -m evaluation --verbose
```

Reports are saved to `evaluation/results/` unless `--no-save` is used.

## Testing

Run all tests:

```bash
pytest -v
```

Run focused suites:

```bash
pytest test/test_chunker.py -v
pytest test/test_embeddings.py -v
pytest test/test_retrieval.py -v
pytest test/test_retrieval_metrics.py -v
```

## Notes

- Embeddings are generated with local sentence-transformer models (CUDA used when available).
- Embedding cache defaults to `.embedding_cache/`.
- Evaluation dataset path defaults to `evaluation/datasets/retrieval_ground_truth.json`.
