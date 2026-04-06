"""CLI for retrieval with single-query and interactive REPL modes."""

import argparse
import logging
import sys
from pathlib import Path

from embeddings.cache import CachedEmbeddingService
from embeddings.models import EmbeddingConfig
from embeddings.service import EmbeddingService
from retrieval.models import RetrievalResponse
from retrieval.service import RetrievalService


def _format_results(response: RetrievalResponse) -> str:
    """Format a RetrievalResponse for terminal output."""
    lines: list[str] = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"Query: {response.query}")
    lines.append(
        f"Results: {response.result_count} (top_k={response.top_k}, "
        f"threshold={response.threshold}, latency={response.latency_ms:.1f}ms)"
    )
    lines.append(f"{'=' * 60}")

    if not response.results:
        lines.append("  No results found.")
        return "\n".join(lines)

    for i, result in enumerate(response.results, 1):
        lines.append(f"\n  [{i}] Score: {result.similarity_score:.4f}")
        lines.append(f"      Source: {result.source_path}")
        if result.document_title:
            lines.append(f"      Title:  {result.document_title}")
        if result.section_path:
            lines.append(f"      Section: {' > '.join(result.section_path)}")
        if result.page_numbers:
            lines.append(f"      Pages: {result.page_numbers}")

        # Truncate text to ~200 chars for display
        text_preview = result.text[:200].replace("\n", " ")
        if len(result.text) > 200:
            text_preview += "..."
        lines.append(f"      Text: {text_preview}")

        if result.summary:
            lines.append(f"      Summary: {result.summary}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def _run_single_query(service: RetrievalService, args: argparse.Namespace) -> None:
    """Run a single query and print results."""
    response = service.retrieve_sync(
        query=args.query,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print(_format_results(response))


def _run_repl(service: RetrievalService, args: argparse.Namespace) -> None:
    """Run an interactive REPL for retrieval testing."""
    print(f"\nFullRag Retrieval REPL (top_k={args.top_k}, threshold={args.threshold})")
    print("Type a query and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Bye.")
            break

        response = service.retrieve_sync(
            query=query,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        print(_format_results(response))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="retrieval",
        description="Search ingested documents by semantic similarity.",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=None,
        help="Single query to run. Omit for interactive REPL mode.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Maximum number of results (default: 5).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Minimum similarity score 0-1 (default: no threshold).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace embedding model (default: BAAI/bge-base-en-v1.5).",
    )
    parser.add_argument(
        "--cache-dir",
        default=".embedding_cache",
        help="Cache directory for query embeddings (default: .embedding_cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache.",
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

    # Build embedding service
    config = EmbeddingConfig(model_name=args.model)
    base_service = EmbeddingService(config=config)

    if args.no_cache:
        embed_service = base_service
    else:
        embed_service = CachedEmbeddingService(
            service=base_service,
            cache_dir=Path(args.cache_dir),
        )

    service = RetrievalService(embedding_service=embed_service, config=config)

    if args.query:
        _run_single_query(service, args)
    else:
        _run_repl(service, args)


if __name__ == "__main__":
    main()