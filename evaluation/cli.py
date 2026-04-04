"""CLI for running retrieval evaluation."""

import argparse
import logging
from pathlib import Path

from embeddings.cache import CachedEmbeddingService
from embeddings.models import EmbeddingConfig
from embeddings.service import EmbeddingService
from evaluation.retrieval_runner import (
    DEFAULT_DATASET_PATH,
    DEFAULT_K_VALUES,
    DEFAULT_OUTPUT_DIR,
    EvaluationRunner,
)
from retrieval.service import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Run retrieval evaluation against a ground-truth dataset.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to ground-truth JSON (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Top-k for retrieval (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Minimum similarity threshold (default: no threshold).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help=f"K values to evaluate metrics at (default: {DEFAULT_K_VALUES}).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for report output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace embedding model (default: BAAI/bge-base-en-v1.5).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".embedding_cache",
        help="Cache directory for query embeddings (default: .embedding_cache).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-query results and enable DEBUG logging.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the report to disk.",
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

    retrieval_service = RetrievalService(
        embedding_service=embed_service, config=config,
    )

    # Run evaluation
    runner = EvaluationRunner(
        retrieval_service=retrieval_service,
        dataset_path=args.dataset,
        top_k=args.top_k,
        threshold=args.threshold,
        k_values=args.k_values,
    )

    report = runner.run()

    # Output
    EvaluationRunner.print_report(report, verbose=args.verbose)

    if not args.no_save:
        filepath = EvaluationRunner.save_report(report, output_dir=args.output_dir)
        print(f"\nReport saved to: {filepath}")


if __name__ == "__main__":
    main()
