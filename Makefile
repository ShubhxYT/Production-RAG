.PHONY: install dev-db dev-backend dev-frontend dev migrate build test clean

# Install all dependencies (backend Python + frontend Node)
install:
	cd backend && uv sync
	cd frontend && npm install

# Start the PostgreSQL + pgvector container only
dev-db:
	docker compose up -d db

# Start the FastAPI development server with hot-reload (requires .env in backend/)
dev-backend:
	cd backend && uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

# Start the Vite frontend dev server (proxies /api to localhost:8001)
dev-frontend:
	cd frontend && npm run dev

# Start all services for local development
# Run in separate terminals for cleaner output:
#   Terminal 1: make dev-db && make dev-backend
#   Terminal 2: make dev-frontend
dev: dev-db
	$(MAKE) dev-backend & $(MAKE) dev-frontend; wait

# Run database migrations (Alembic)
migrate:
	cd backend && uv run alembic upgrade head

# Build frontend production bundle
build:
	cd frontend && npm run build

# Run backend test suite
test:
	cd backend && uv run pytest -v

# Start Streamlit admin dashboard
streamlit:
	cd backend && uv run streamlit run streamlit_app/app.py

# Remove generated artifacts
clean:
	rm -rf frontend/dist
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
