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


def get_database_url() -> str:
    """Return the database URL from environment.

    Defaults to the local Docker PostgreSQL from pgvector.yaml
    if DATABASE_URL is not set.

    Returns:
        SQLAlchemy-compatible database URL.
    """
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://fullrag:fullrag@localhost:5432/fullrag",
    )


def get_generation_model() -> str:
    """Return the generation model name from environment.

    Defaults to gemini-2.5-flash.
    """
    return os.environ.get("GENERATION_MODEL", "gemini-2.5-flash")


def get_generation_temperature() -> float:
    """Return the generation temperature from environment.

    Defaults to 0.3.
    """
    return float(os.environ.get("GENERATION_TEMPERATURE", "0.3"))


def get_generation_max_tokens() -> int:
    """Return the max output tokens for generation from environment.

    Defaults to 2048.
    """
    return int(os.environ.get("GENERATION_MAX_TOKENS", "2048"))


def get_generation_provider() -> str:
    """Return the generation provider name from environment.

    Supported: 'gemini', 'cerebras'. Defaults to 'gemini'.
    """
    return os.environ.get("GENERATION_PROVIDER", "gemini")


def get_cerebras_base_url() -> str:
    """Return the Cerebras API base URL.

    Defaults to the official Cerebras OpenAI-compatible endpoint.
    """
    return os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")


def get_log_level() -> str:
    """Return the log level from environment.

    Defaults to INFO. Supported: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    return os.environ.get("LOG_LEVEL", "INFO").upper()


def get_log_format() -> str:
    """Return the log format from environment.

    Supported: 'json' (structured), 'text' (human-readable).
    Defaults to 'json'.
    """
    return os.environ.get("LOG_FORMAT", "json").lower()


def get_log_file() -> str | None:
    """Return the log file path from environment.

    If not set, logs go to stderr only. If set, logs are also
    written to a rotating file.
    """
    return os.environ.get("LOG_FILE") or None
