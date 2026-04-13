FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies into .venv
RUN uv sync --frozen --no-dev

# Copy application source
COPY . .

# Add .venv binaries to PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

# Run DB migrations then start the API server
CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8001"]
