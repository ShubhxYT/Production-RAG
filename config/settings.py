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
