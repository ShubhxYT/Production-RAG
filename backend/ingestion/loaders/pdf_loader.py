"""PDF document loader using Docling."""

import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

from ingestion.models import Document

logger = logging.getLogger(__name__)


class PdfLoader:
    """Load and convert PDF files to markdown using Docling."""

    def load(self, file_path: Path, output_dir: Path) -> Document:
        """Load a PDF file and convert it to a Document with markdown content.

        Args:
            file_path: Path to the PDF file.
            output_dir: Directory to save extracted images.

        Returns:
            Document with raw_content populated as markdown text.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc_output_dir = output_dir / file_path.stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            pipeline_opts = PdfPipelineOptions(generate_picture_images=True)
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_opts
                    )
                }
            )
            result = converter.convert(str(file_path))

            out_md = doc_output_dir / f"{file_path.stem}.md"
            result.document.save_as_markdown(
                out_md,
                artifacts_dir=Path("images"),
                image_mode=ImageRefMode.REFERENCED,
            )
            md_text = out_md.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to convert PDF: %s", file_path)
            raise

        logger.info("Converted PDF: %s", file_path.name)

        return Document(
            source_path=str(file_path),
            title=file_path.stem,
            format="pdf",
            raw_content=md_text,
            metadata={"output_dir": str(doc_output_dir)},
        )
