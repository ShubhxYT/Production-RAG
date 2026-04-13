# FullRag-Plus

A production-ready Retrieval-Augmented Generation (RAG) system combining a FastAPI Python backend with a React TypeScript frontend, backed by PostgreSQL + pgvector for semantic search.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FullRag-Plus                        │
│                                                     │
│  ┌──────────────────┐     ┌─────────────────────┐   │
│  │  React Frontend  │     │  Streamlit Admin    │   │
│  │  (port 5173/80)  │     │  (port 8501)        │   │
│  └────────┬─────────┘     └─────────────────────┘   │
│           │ /api/* requests                          │
│           ▼                                          │
│  ┌──────────────────┐                               │
│  │  FastAPI Backend │                               │
│  │  (port 8001)     │                               │
│  └────┬─────────────┘                               │
│       │              │                              │
│       ▼              ▼                              │
│  ┌─────────┐   ┌──────────────┐                    │
│  │pgvector │   │ LLM Provider │                    │
│  │  (5432) │   │ Gemini/Groq  │                    │
│  └─────────┘   └──────────────┘                    │
└─────────────────────────────────────────────────────┘
```

**Request flow:** User query → React UI → FastAPI `/query` → pgvector similarity search → LLM generation (Gemini or Groq) → answer with sources returned to UI.

## Repository Structure

```
FullRag-Plus/
├── backend/                    # Python FastAPI backend (fullrag-plus-backend)
│   ├── api/                    # FastAPI app: main.py + routes/
│   │   └── routes/             #   health, documents, query, evaluation
│   ├── config/                 # Settings loaded from .env
│   ├── database/               # SQLAlchemy models, repository, seed
│   ├── embeddings/             # Embedding service (sentence-transformers)
│   ├── generation/             # LLM generation pipeline (Gemini/Groq)
│   ├── ingestion/              # Document ingestion and chunking pipeline
│   ├── pipeline/               # RAG orchestration (retrieval + generation)
│   ├── retrieval/              # pgvector similarity search
│   ├── evaluation/             # Continuous evaluation loop + metrics
│   ├── observability/          # Structured logging + tracing
│   ├── streamlit_app/          # Admin debug dashboard
│   ├── alembic/                # Database migrations
│   ├── data/                   # Transcript fallback (data/transcript.md)
│   ├── test/                   # pytest test suite
│   ├── pyproject.toml          # Python dependencies (managed by uv)
│   ├── uv.lock                 # Locked Python dependencies
│   ├── alembic.ini             # Alembic migration config
│   └── Dockerfile              # Backend container
├── frontend/                   # React + TypeScript frontend (fullrag-plus)
│   ├── src/
│   │   ├── main.tsx            # App entry point
│   │   ├── App.tsx             # Routes: Home, Results, Library
│   │   ├── lib/                # API client (axios) + TypeScript types
│   │   ├── pages/              # Home, Results, Library pages
│   │   └── components/         # Layout, search, results, library components
│   ├── public/                 # Static assets
│   ├── index.html              # HTML shell (dark mode, Manrope/Inter fonts)
│   ├── package.json
│   ├── vite.config.ts          # Dev server port 5173, proxy /api → :8001
│   ├── tailwind.config.js      # Material Design 3 dark theme tokens
│   ├── nginx.conf              # Production nginx (SPA + /api proxy)
│   └── Dockerfile              # Multi-stage: node:20-alpine + nginx:alpine
├── .env.example                # All environment variable templates
├── .gitignore
├── docker-compose.yml          # PostgreSQL + backend + frontend (nginx)
├── Makefile                    # Developer convenience targets
└── README.md
```

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | [python.org](https://www.python.org) |
| uv | latest | `pip install uv` |
| Node.js | 20+ | [nodejs.org](https://nodejs.org) |
| Docker | 20+ | [docker.com](https://www.docker.com) |
| Docker Compose | v2 | Included with Docker Desktop |

**API Keys required (at least one LLM provider):**
- **Google Gemini:** [aistudio.google.com](https://aistudio.google.com) → Create API key
- **Groq:** [console.groq.com](https://console.groq.com) → Create API key

## Quick Start (Local Development)

```bash
# 1. Clone and enter the repo
git clone <repo-url> FullRag-Plus
cd FullRag-Plus

# 2. Set up environment variables
cp .env.example .env
# Edit .env and fill in GEMINI_API_KEY and/or GROQ_API_KEY

# 3. Install all dependencies
make install

# 4. Start the database
make dev-db

# 5. Run database migrations
make migrate

# 6. Start both dev servers (in separate terminals for best experience)
# Terminal 1:
make dev-backend    # FastAPI at http://localhost:8001

# Terminal 2:
make dev-frontend   # React at http://localhost:5173
```

Open **http://localhost:5173** in your browser.

## Docker Production

```bash
# 1. Set up environment variables
cp .env.example .env
# Edit .env — set real API keys

# 2. Build and start all services
docker compose up --build

# Services started:
#   PostgreSQL + pgvector  →  localhost:5432
#   FastAPI backend        →  localhost:8001
#   React frontend (nginx) →  localhost:80
```

Open **http://localhost** in your browser.

The backend container automatically runs `alembic upgrade head` on startup before launching the API server.

## Make Targets

| Command | Description |
|---------|-------------|
| `make install` | Install Python (`uv sync`) and Node (`npm install`) dependencies |
| `make dev-db` | Start PostgreSQL container in the background |
| `make dev-backend` | Start FastAPI with hot-reload on port 8001 |
| `make dev-frontend` | Start Vite dev server on port 5173 |
| `make dev` | Start all (db + backend + frontend concurrently) |
| `make migrate` | Run Alembic migrations (`alembic upgrade head`) |
| `make build` | Build frontend production bundle to `frontend/dist/` |
| `make test` | Run pytest test suite in `backend/test/` |
| `make streamlit` | Start Streamlit admin dashboard on port 8501 |
| `make clean` | Remove `frontend/dist/` and `__pycache__/` directories |

## API Reference

Base URL (local dev): `http://localhost:8001`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns DB connection status |
| `GET` | `/documents` | List all ingested documents with chunk counts |
| `POST` | `/query` | Submit a RAG query — returns answer + sources |
| `POST` | `/feedback` | Submit user feedback on a query result |
| `GET` | `/evaluation/summary` | Get evaluation metrics summary |

### Query Request/Response

```json
// POST /query
{
  "question": "What is the main topic of the document?",
  "top_k": 5,
  "prompt_variant": "default"
}

// Response
{
  "answer": "The document discusses...",
  "sources": [{ "chunk_id": "...", "content": "...", "score": 0.92 }],
  "latency": { "total_ms": 1240 },
  "token_usage": { "input": 512, "output": 128 },
  "prompt_version": "default",
  "query_log_id": "uuid"
}
```

## Environment Variables

All variables are read from `.env` in the project root (Docker Compose) or `backend/.env` (local dev).

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `GEMINI_API_KEY` | — | Yes* | Google Gemini API key |
| `GROQ_API_KEY` | — | Yes* | Groq API key (*one LLM key required) |
| `DATABASE_URL` | `postgresql+psycopg://fullrag:fullrag@localhost:5432/fullrag` | Yes | PostgreSQL connection string |
| `GENERATION_PROVIDER` | `gemini` | No | Active LLM provider: `gemini` or `groq` |
| `GENERATION_MODEL` | `gemini-2.5-flash` | No | Gemini model name |
| `GENERATION_TEMPERATURE` | `0.3` | No | LLM sampling temperature |
| `GENERATION_MAX_TOKENS` | `2048` | No | Maximum output tokens |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | No | Groq model name |
| `LOG_LEVEL` | `INFO` | No | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FORMAT` | `json` | No | `json` or `text` |
| `CACHE_TTL_SECONDS` | `3600` | No | Response cache TTL |
| `RESPONSE_CACHE_ENABLED` | `true` | No | Toggle response cache |
| `CONTINUOUS_EVAL_ENABLED` | `true` | No | Toggle background evaluation loop |
| `TRANSCRIPT_FALLBACK_ENABLED` | `true` | No | Enable transcript file fallback |
| `TRANSCRIPT_PATH` | `data/transcript.md` | No | Path to transcript fallback file |

## Document Ingestion (CLI)

After setup, ingest documents through the backend CLI:

```bash
cd backend

# Ingest a PDF or text file
uv run python -m ingestion.cli ingest path/to/document.pdf

# Check ingestion status
uv run python -m ingestion.cli status

# Generate and store embeddings
uv run python -m embeddings.cli embed --all
```

## Streamlit Admin Dashboard

A debug/admin dashboard is available via Streamlit:

```bash
make streamlit
# Opens at http://localhost:8501
```

Features: document browser, query tester, evaluation metrics viewer, embedding cache inspector.

## Running Tests

```bash
make test

# Or with coverage:
cd backend && uv run pytest -v --tb=short
```

Test files are in `backend/test/` covering: database, embeddings, chunker, API endpoints, and continuous evaluation.

## Troubleshooting

**Database connection refused:**
```bash
make dev-db          # Ensure container is running
docker ps            # Verify fullrag-plus-postgres is up
```

**`alembic upgrade head` fails:**
```bash
# Check DATABASE_URL in .env points to running database
cd backend && uv run alembic current
```

**Frontend cannot reach API:**
- Local dev: Vite proxies `/api` → `http://localhost:8001` (check backend is running)
- Docker: nginx proxies `/api` → `http://backend:8001` (check `docker compose ps`)

**Large Docker build (torch/sentence-transformers):**
The backend image includes PyTorch for sentence-transformers embeddings. Initial build downloads ~2–3 GB. Subsequent builds use the Docker layer cache.
