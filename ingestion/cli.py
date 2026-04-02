"""CLI for the ingestion pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ingestion",
        description="Ingest documents into the FullRag pipeline.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a file or directory to ingest.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results",
        help="Output directory for processed documents (default: results).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that have already been processed.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    pipeline = IngestionPipeline(output_dir=args.output)

    if input_path.is_dir():
        documents = pipeline.ingest_directory(
            input_path, skip_existing=args.skip_existing
        )
    elif input_path.is_file():
        doc = pipeline.ingest_file(
            input_path, skip_existing=args.skip_existing
        )
        documents = [doc] if doc else []
    else:
        print(f"Error: path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    total_elements = sum(len(d.elements) for d in documents)
    element_counts: dict[str, int] = {}
    for doc in documents:
        for el in doc.elements:
            key = el.type.value
            element_counts[key] = element_counts.get(key, 0) + 1

    print(f"\n{'=' * 50}")
    print("Ingestion Summary")
    print(f"{'=' * 50}")
    print(f"Files processed: {len(documents)}")
    print(f"Total elements:  {total_elements}")
    if element_counts:
        print("\nElement breakdown:")
        for etype, count in sorted(element_counts.items()):
            print(f"  {etype:15s} {count}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
