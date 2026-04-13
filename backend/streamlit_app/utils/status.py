"""System health checks for the FullRag Streamlit debug UI."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path for imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def check_db() -> tuple[bool, str]:
    """Check if the PostgreSQL/pgvector database is reachable."""
    try:
        from database.connection import get_session
        session = get_session()
        session.execute(__import__("sqlalchemy").text("SELECT 1"))
        session.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def check_gemini() -> tuple[bool, str]:
    """Check if the Gemini API key is configured."""
    try:
        from config.settings import get_gemini_api_key, get_generation_model
        key = get_gemini_api_key()
        model = get_generation_model()
        return True, f"Key found — model: {model}"
    except Exception as e:
        return False, str(e)


def check_groq() -> tuple[bool, str]:
    """Check if the Groq API key is configured."""
    try:
        from config.settings import get_groq_api_key, get_groq_base_url
        key = get_groq_api_key()
        url = get_groq_base_url()
        return True, f"Key found — endpoint: {url}"
    except Exception as e:
        return False, str(e)


def check_gpu() -> tuple[bool, str]:
    """Check GPU/CUDA availability for embeddings."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            return True, f"CUDA — {name} ({mem} GB)"
        return False, "No CUDA — using CPU"
    except Exception as e:
        return False, f"torch error: {e}"


def get_embedding_model_name() -> str:
    """Return the configured embedding model name."""
    try:
        from embeddings.models import EmbeddingConfig
        return EmbeddingConfig().model_name
    except Exception:
        return "Unknown"
