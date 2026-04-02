"""Staging: serialize and deserialize Documents as JSON files."""

import logging
from pathlib import Path

from ingestion.models import Document

logger = logging.getLogger(__name__)


def stage_document(
    document: Document, staging_dir: str = "staging"
) -> Path:
    """Save a Document as a JSON file in the staging directory.

    Args:
        document: The Document to serialize.
        staging_dir: Directory for staged JSON files.

    Returns:
        Path to the saved JSON file.
    """
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)

    out_file = staging_path / f"{document.id}.json"
    out_file.write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Staged: %s -> %s", document.id, out_file)
    return out_file


def load_staged_document(json_path: Path) -> Document:
    """Load a Document from a staged JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Deserialized Document object.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Staged file not found: {json_path}")

    json_text = json_path.read_text(encoding="utf-8")
    return Document.model_validate_json(json_text)


def stage_all(
    documents: list[Document], staging_dir: str = "staging"
) -> list[Path]:
    """Stage multiple Documents to JSON files.

    Args:
        documents: Documents to serialize.
        staging_dir: Directory for staged JSON files.

    Returns:
        List of paths to saved JSON files.
    """
    return [stage_document(doc, staging_dir) for doc in documents]
