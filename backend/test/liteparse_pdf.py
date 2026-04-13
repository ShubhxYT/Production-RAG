# Install: uv add liteparse
# Note: liteparse wraps a Node.js CLI (@llamaindex/liteparse).
# Node.js >=18 and npm must be in PATH; the CLI is auto-installed on first run.
# No native GPU/CUDA support — extraction runs via PDF.js (CPU/JS).
from pathlib import Path
from typing import List
print("Loading imports...")
from langchain_core.documents import Document
print("langchain_core imported")
from liteparse import LiteParse
print("liteparse imported")



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

    parser = LiteParse(cli_path="npx --yes @llamaindex/liteparse")
    print(f"Parsing {pdf_path}...")
    result = parser.parse(pdf_path)
    print(f"Parsing completed for {pdf_path}")

    stem = Path(pdf_path).stem
    print(f"Saving result to {results_path / f'{stem}.md'}...")
    out_file = results_path / f"{stem}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result.text)
    print(f"Saved: {out_file}")

    return [Document(page_content=result.text, metadata={"source": pdf_path})]


def convert_directory(pdf_dir: str, results_dir: str) -> List[Document]:
    """Convert all PDFs in pdf_dir and save each as a markdown file in results_dir.

    Skips any PDF whose output .md file already exists in results_dir.

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

    parser = LiteParse(cli_path="npx --yes @llamaindex/liteparse")
    all_docs: List[Document] = []
    for p in pdf_files:
        result = parser.parse(str(p))
        out_file = results_path / f"{p.stem}.md"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.text)
        print(f"Saved: {out_file}")
        all_docs.append(Document(page_content=result.text, metadata={"source": str(p)}))

    return all_docs


if __name__ == "__main__":
    # Example usage
    convert_single("data/test_test.pdf", "results")
    # convert_directory("data", "results")
