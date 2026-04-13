"""Auto-cleanup utilities for temporary user data."""

import shutil
from pathlib import Path

USER_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "user_data"


def clean_user_data() -> None:
    """Delete and recreate the user_data/ directory on every app startup."""
    if USER_DATA_DIR.exists():
        shutil.rmtree(USER_DATA_DIR)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_user_data_dir() -> Path:
    """Return the user_data directory path, creating it if needed."""
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return USER_DATA_DIR
