"""CLI for running retrieval and generation evaluation."""

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


def _run_retrieval(args: argparse.Namespace) -> None:
    """Run retrieval evaluation."""
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

    runner = EvaluationRunner(
        retrieval_service=retrieval_service,
        dataset_path=args.dataset,
        top_k=args.top_k,
        threshold=args.threshold,
        k_values=args.k_values,
    )

    report = runner.run()
    EvaluationRunner.print_report(report, verbose=args.verbose)

    if not args.no_save:
        filepath = EvaluationRunner.save_report(report, output_dir=args.output_dir)
        print(f"\nReport saved to: {filepath}")


def _run_generation(args: argparse.Namespace) -> None:
    """Run generation evaluation."""
    from config.settings import get_generation_provider as get_provider_name

    from evaluation.generation_judges import JudgePanel
    from evaluation.generation_runner import (
        DEFAULT_GENERATION_DATASET_PATH,
        GenerationEvaluationRunner,
    )
    from pipeline.rag import RAGPipeline

    # Build RAG pipeline
    pipeline = RAGPipeline(provider_name=get_provider_name())

    # Build judge panel
    judge_provider = args.judge_provider or get_provider_name()
    panel = JudgePanel.default_panel(provider_name=judge_provider)

    # Dataset path
    dataset_path = args.generation_dataset or DEFAULT_GENERATION_DATASET_PATH

    runner = GenerationEvaluationRunner(
        rag_pipeline=pipeline,
        judge_panel=panel,
        dataset_path=dataset_path,
        top_k=args.top_k,
    )

    report = runner.run()
    GenerationEvaluationRunner.print_report(report, verbose=args.verbose)

    if not args.no_save:
        filepath = GenerationEvaluationRunner.save_report(
            report, output_dir=args.output_dir,
        )
        print(f"\nReport saved to: {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Run retrieval and/or generation evaluation.",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "generation", "all"],
        default="retrieval",
        help="Evaluation mode (default: retrieval).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to retrieval ground-truth JSON (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--generation-dataset",
        type=Path,
        default=None,
        help="Path to generation ground-truth JSON (default: built-in).",
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
        "--judge-provider",
        default=None,
        help="LLM provider for judges (default: same as generation provider).",
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

    if args.mode in ("retrieval", "all"):
        _run_retrieval(args)

    if args.mode in ("generation", "all"):
        _run_generation(args)


if __name__ == "__main__":
    main()
