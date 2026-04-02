"""Ingestion pipeline orchestrator."""

import logging
from pathlib import Path

from ingestion.loaders.docx_loader import DocxLoader
from ingestion.loaders.html_loader import HtmlLoader
from ingestion.loaders.markdown_loader import MarkdownLoader
from ingestion.loaders.pdf_loader import PdfLoader
from ingestion.models import Document
from ingestion.restructurer import restructure

logger = logging.getLogger(__name__)

LOADER_REGISTRY: dict[str, type] = {
    ".pdf": PdfLoader,
    ".docx": DocxLoader,
    ".html": HtmlLoader,
    ".htm": HtmlLoader,
    ".md": MarkdownLoader,
}


class IngestionPipeline:
    """Orchestrate document ingestion: load, restructure, return Documents."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._loaders: dict[str, object] = {}

    def _get_loader(self, extension: str):
        """Get or create a cached loader instance for the given extension."""
        if extension not in self._loaders:
            loader_cls = LOADER_REGISTRY.get(extension)
            if loader_cls is None:
                return None
            self._loaders[extension] = loader_cls()
        return self._loaders[extension]

    def ingest_file(
        self, file_path: Path, skip_existing: bool = False
    ) -> Document | None:
        """Ingest a single file: load and restructure into Elements.

        Args:
            file_path: Path to the document file.
            skip_existing: Skip if the output directory already exists.

        Returns:
            Populated Document with elements, or None if skipped/failed.
        """
        ext = file_path.suffix.lower()
        loader = self._get_loader(ext)

        if loader is None:
            logger.debug("Unsupported file type: %s", file_path.name)
            return None

        if skip_existing:
            doc_output = self.output_dir / file_path.stem
            if doc_output.exists():
                logger.info("Skipping (exists): %s", file_path.name)
                return None

        try:
            doc = loader.load(file_path, self.output_dir)
            doc.elements = restructure(doc.raw_content)
            return doc
        except Exception:
            logger.exception("Failed to process: %s", file_path.name)
            return None

    def ingest_directory(
        self, input_dir: Path, skip_existing: bool = False
    ) -> list[Document]:
        """Ingest all supported files in a directory.

        Args:
            input_dir: Directory containing documents to ingest.
            skip_existing: Skip files whose output directory already exists.

        Returns:
            List of successfully processed Document objects.
        """
        if not input_dir.is_dir():
            logger.error("Not a directory: %s", input_dir)
            return []

        supported = set(LOADER_REGISTRY.keys())
        files = sorted(
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        )

        if not files:
            logger.info("No supported files in %s", input_dir)
            return []

        documents: list[Document] = []
        skipped = 0
        failed = 0

        for file_path in files:
            if skip_existing and (self.output_dir / file_path.stem).exists():
                logger.info("Skipping (exists): %s", file_path.name)
                skipped += 1
                continue

            doc = self.ingest_file(file_path)
            if doc is not None:
                documents.append(doc)
            else:
                failed += 1

        total_elements = sum(len(d.elements) for d in documents)
        logger.info(
            "Ingestion: %d processed, %d skipped, %d failed, %d elements",
            len(documents),
            skipped,
            failed,
            total_elements,
        )
        return documents
