"""Environment-based settings loader."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (three levels up from this file: config -> backend -> root)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
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


def get_groq_api_key() -> str:
    """Return the Groq API key from environment.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
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

    Supported: 'gemini', 'groq'. Defaults to 'gemini'.
    """
    return os.environ.get("GENERATION_PROVIDER", "gemini")


def get_groq_base_url() -> str:
    """Return the Groq API base URL.

    Defaults to the official Groq OpenAI-compatible endpoint.
    """
    return os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


def get_groq_model() -> str:
    """Return the Groq model name from environment.

    Defaults to llama-3.3-70b-versatile. Override with GROQ_MODEL env var.
    """
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_mistral_api_key() -> str:
    """Return the Mistral API key from environment.

    Raises:
        RuntimeError: If MISTRAL_API_KEY is not set.
    """
    key = os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set. Add it to your .env file."
        )
    return key


def get_mistral_model() -> str:
    """Return the Mistral model name for enrichment from environment.

    Defaults to mistral-small-latest.
    """
    return os.environ.get("MISTRAL_MODEL", "mistral-small-latest")


def get_enrichment_provider() -> str:
    """Return the enrichment provider name from environment.

    Supported: 'gemini', 'mistral'. Defaults to 'gemini'.
    """
    return os.environ.get("ENRICHMENT_PROVIDER", "gemini")


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


def get_cache_ttl_seconds() -> int:
    """Return the response cache TTL in seconds.

    Defaults to 3600 (1 hour).
    """
    return int(os.environ.get("CACHE_TTL_SECONDS", "3600"))


def get_cache_max_size() -> int:
    """Return the maximum number of entries in the response cache.

    Defaults to 256.
    """
    return int(os.environ.get("CACHE_MAX_SIZE", "256"))


def get_response_cache_enabled() -> bool:
    """Return whether the response cache is enabled.

    Defaults to True.
    """
    return os.environ.get("RESPONSE_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")


def get_continuous_eval_enabled() -> bool:
    """Return whether continuous evaluation is enabled.

    Defaults to True.
    """
    return os.environ.get("CONTINUOUS_EVAL_ENABLED", "true").lower() in ("true", "1", "yes")


def get_eval_schedule_interval_hours() -> int:
    """Return the evaluation schedule interval in hours.

    Defaults to 24 (daily).
    """
    return int(os.environ.get("EVAL_SCHEDULE_INTERVAL_HOURS", "24"))


def get_transcript_fallback_enabled() -> bool:
    """Return whether transcript fallback is enabled.

    When True, queries that return insufficient context fall back to
    answering from the raw transcript file. Defaults to True.
    """
    return os.environ.get("TRANSCRIPT_FALLBACK_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )


def get_transcript_path() -> str:
    """Return the path to the transcript file used as fallback context.

    Defaults to 'data/transcript.md'. Override with TRANSCRIPT_PATH env var.
    """
    return os.environ.get("TRANSCRIPT_PATH", "data/transcript.md")
