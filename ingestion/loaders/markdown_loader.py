"""Markdown document loader (passthrough)."""

import logging
import re
from pathlib import Path

from ingestion.models import Document

logger = logging.getLogger(__name__)


class MarkdownLoader:
    """Load markdown files as-is into the Document model."""

    def load(self, file_path: Path, output_dir: Path) -> Document:
        """Load a markdown file into a Document.

        Args:
            file_path: Path to the markdown file.
            output_dir: Directory for output artifacts (unused for markdown).

        Returns:
            Document with raw_content as the markdown file text.

        Raises:
            FileNotFoundError: If the markdown file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        md_text = file_path.read_text(encoding="utf-8")

        # Extract title from the first ATX heading
        title = None
        match = re.match(r"^#\s+(.+)$", md_text, re.MULTILINE)
        if match:
            title = match.group(1).strip()

        logger.info("Loaded markdown: %s", file_path.name)

        return Document(
            source_path=str(file_path),
            title=title or file_path.stem,
            format="md",
            raw_content=md_text,
        )
