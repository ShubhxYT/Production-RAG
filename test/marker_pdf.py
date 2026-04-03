# Install: uv add marker-pdf
# Note: marker-pdf code is GPL-3.0; model weights use AI Pubs Open Rail-M license.
# Requires PyTorch. Models (~several GBs) are downloaded on first run.
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def convert_single(pdf_path: str, results_dir: str) -> List[Document]:
    """Convert a single PDF and save it as a markdown file in results_dir.

    Args:
        pdf_path: Path to the PDF file.
        results_dir: Path to the directory where the markdown result will be saved.

    Returns:
        List of LangChain Document objects for the converted PDF.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    markdown, _, _ = text_from_rendered(rendered)

    stem = Path(pdf_path).stem
    out_file = results_path / f"{stem}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Saved: {out_file}")

    return [Document(page_content=markdown, metadata={"source": pdf_path})]


def convert_directory(pdf_dir: str, results_dir: str) -> List[Document]:
    """Convert all PDFs in pdf_dir and save each as a markdown file in results_dir.

    Skips any PDF whose output .md file already exists in results_dir.
    A single PdfConverter instance is reused across all files to avoid
    reloading the deep learning models for each PDF.

    Args:
        pdf_dir: Path to the directory containing PDF files.
        results_dir: Path to the directory where markdown results will be saved.

    Returns:
        List of LangChain Document objects for all converted PDFs.
    """
    pdf_dir_path = Path(pdf_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    all_pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
    if not all_pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []

    pdf_files = []
    for p in all_pdf_files:
        out_file = results_path / f"{p.stem}.md"
        if out_file.exists():
            print(f"Skipping (already converted): {p.name}")
        else:
            pdf_files.append(p)

    if not pdf_files:
        print("All PDFs already converted, nothing to do.")
        return []

    # Load models once and reuse the converter for all files
    converter = PdfConverter(artifact_dict=create_model_dict())

    all_docs: List[Document] = []
    for p in pdf_files:
        rendered = converter(str(p))
        markdown, _, _ = text_from_rendered(rendered)

        out_file = results_path / f"{p.stem}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Saved: {out_file}")

        all_docs.append(Document(page_content=markdown, metadata={"source": str(p)}))

    return all_docs


if __name__ == "__main__":
    # Example usage
    convert_single("data/test_test.pdf", "results")
    # convert_directory("data/pdfs", "results")
    pass
