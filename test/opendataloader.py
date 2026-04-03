# Requires Java 11+ available on the system PATH.
# Install: uv add langchain-opendataloader-pdf
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader


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

    loader = OpenDataLoaderPDFLoader(file_path=pdf_path, format="markdown")
    docs = loader.load()

    stem = Path(pdf_path).stem
    out_file = results_path / f"{stem}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.page_content)
            f.write("\n\n")
    print(f"Saved: {out_file}")

    return docs


def convert_directory(pdf_dir: str, results_dir: str) -> List[Document]:
    """Convert all PDFs in pdf_dir and save each as a markdown file in results_dir.

    Skips any PDF whose output .md file already exists in results_dir.
    All remaining PDFs are batched into a single loader call to avoid
    spawning a JVM process per file.

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
            pdf_files.append(str(p))

    if not pdf_files:
        print("All PDFs already converted, nothing to do.")
        return []

    loader = OpenDataLoaderPDFLoader(file_path=pdf_files, format="markdown")
    docs = loader.load()

    # Group docs by source and write one markdown file per PDF
    by_source: dict[str, List[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        by_source.setdefault(source, []).append(doc)

    for source, source_docs in by_source.items():
        stem = Path(source).stem
        out_file = results_path / f"{stem}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            for doc in source_docs:
                f.write(doc.page_content)
                f.write("\n\n")
        print(f"Saved: {out_file}")

    return docs


if __name__ == "__main__":
    # Example usage
    # convert_single("data/test_test.pdf", "results")
    convert_directory("data", "results")
    pass
