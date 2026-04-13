"""CLI for the ingestion pipeline."""

import argparse
import logging
import sys
import time
from pathlib import Path

from ingestion.pipeline import IngestionPipeline


def _run_ingest(args: argparse.Namespace) -> None:
    """Run the ingest subcommand."""
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

    # Print element summary
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

    # Print chunk summary
    total_chunks = sum(len(d.chunks) for d in documents)
    if total_chunks > 0:
        all_token_counts = [
            c.token_count for d in documents for c in d.chunks
        ]
        min_tokens = min(all_token_counts)
        max_tokens = max(all_token_counts)
        avg_tokens = sum(all_token_counts) / len(all_token_counts)

        chunk_element_counts: dict[str, int] = {}
        for doc in documents:
            for chunk in doc.chunks:
                for et in chunk.element_types:
                    key = et.value
                    chunk_element_counts[key] = (
                        chunk_element_counts.get(key, 0) + 1
                    )

        print(f"\n{'─' * 50}")
        print("Chunking Summary")
        print(f"{'─' * 50}")
        print(f"Total chunks:    {total_chunks}")
        print(f"Token counts:    min={min_tokens}  max={max_tokens}  avg={avg_tokens:.0f}")
        if chunk_element_counts:
            print("\nElement types in chunks:")
            for etype, count in sorted(chunk_element_counts.items()):
                print(f"  {etype:15s} {count}")

    print(f"{'=' * 50}")


def _run_enrich(args: argparse.Namespace) -> None:
    """Run the enrich subcommand."""
    from generation.llm_service import get_enrichment_provider
    from generation.models import LLMConfig
    from ingestion.enrichment import enrich_chunks
    from ingestion.staging import load_staged_document, stage_document

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not input_path.is_dir():
        print(
            f"Error: staging directory does not exist: {args.input}",
            file=sys.stderr,
        )
        sys.exit(1)

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load staged documents
    documents = []
    for jf in json_files:
        try:
            doc = load_staged_document(jf)
            documents.append(doc)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to load: %s", jf.name
            )

    # Collect all chunks across documents
    all_chunks = [c for doc in documents for c in doc.chunks]
    if not all_chunks:
        print("No chunks found in staged documents.", file=sys.stderr)
        sys.exit(1)

    # Set up LLM service
    config = LLMConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )
    provider = get_enrichment_provider(config=config)

    # Enrich all chunks
    start_time = time.perf_counter()
    enrich_chunks(all_chunks, provider, batch_delay=args.delay)
    elapsed = time.perf_counter() - start_time

    # Count results
    enriched = sum(1 for c in all_chunks if c.summary)
    total = len(all_chunks)

    # Re-save documents with enriched chunks
    saved = 0
    for doc in documents:
        stage_document(doc, staging_dir=str(output_path))
        saved += 1

    # Print summary
    print(f"\n{'=' * 50}")
    print("Enrichment Summary")
    print(f"{'=' * 50}")
    print(f"Documents:         {len(documents)}")
    print(f"Total chunks:      {total}")
    print(f"Enriched:          {enriched}")
    print(f"Skipped/Failed:    {total - enriched}")
    print(f"Model:             {config.model_name}")
    print(f"Elapsed time:      {elapsed:.1f}s")
    print(f"Output directory:  {output_path}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ingestion",
        description="FullRag ingestion pipeline: ingest documents and enrich chunks.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- ingest subcommand --
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents from a file or directory."
    )
    ingest_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a file or directory to ingest.",
    )
    ingest_parser.add_argument(
        "--output",
        "-o",
        default="results",
        help="Output directory for processed documents (default: results).",
    )
    ingest_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that have already been processed.",
    )

    # -- enrich subcommand --
    enrich_parser = subparsers.add_parser(
        "enrich", help="Enrich staged document chunks with LLM-generated metadata."
    )
    enrich_parser.add_argument(
        "--input",
        "-i",
        default="staging",
        help="Path to staging directory with Document JSON files (default: staging).",
    )
    enrich_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for enriched JSONs (default: same as input).",
    )
    enrich_parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash).",
    )
    enrich_parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3).",
    )
    enrich_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max output tokens (default: 1024).",
    )
    enrich_parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between API calls for rate limiting (default: 0.5).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "enrich":
        _run_enrich(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
