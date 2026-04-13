"""HTML document loader using BeautifulSoup and markdownify."""

import logging
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from ingestion.models import Document

logger = logging.getLogger(__name__)


class HtmlLoader:
    """Load and convert HTML files to markdown."""

    STRIP_TAGS = ["script", "style", "nav", "footer", "header"]

    def load(self, file_path: Path, output_dir: Path) -> Document:
        """Load an HTML file and convert it to a Document with markdown content.

        Args:
            file_path: Path to the HTML file.
            output_dir: Directory for output artifacts.

        Returns:
            Document with raw_content populated as markdown text.

        Raises:
            FileNotFoundError: If the HTML file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {file_path}")

        html_text = file_path.read_text(encoding="utf-8", errors="replace")

        soup = BeautifulSoup(html_text, "html.parser")

        # Extract metadata before stripping tags
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        meta_info: dict = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name", meta.get("property", ""))
            content = meta.get("content", "")
            if name and content:
                meta_info[name] = content

        # Remove unwanted tags
        for tag_name in self.STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Convert to markdown
        md_text = md(str(soup), heading_style="ATX")
        md_text = md_text.strip()

        doc_output_dir = output_dir / file_path.stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        out_md = doc_output_dir / f"{file_path.stem}.md"
        out_md.write_text(md_text, encoding="utf-8")
        logger.info("Converted HTML: %s", file_path.name)

        return Document(
            source_path=str(file_path),
            title=title or file_path.stem,
            format="html",
            raw_content=md_text,
            metadata={"output_dir": str(doc_output_dir), **meta_info},
        )
