"""Generation evaluation runner — runs RAG pipeline and scores with LLM judges."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from evaluation.generation_judges import JudgePanel
from evaluation.models import (
    GenerationEvaluationReport,
    GenerationGroundTruth,
    GenerationQueryResult,
    JudgeDimension,
)
from pipeline.rag import RAGPipeline

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_DATASET_PATH = Path(
    "evaluation/datasets/generation_ground_truth.json"
)
DEFAULT_OUTPUT_DIR = Path("evaluation/results")


class GenerationEvaluationRunner:
    """Runs end-to-end generation evaluation.

    For each query in the dataset, runs the RAG pipeline, then
    evaluates the answer with the judge panel.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        judge_panel: JudgePanel,
        dataset_path: Path = DEFAULT_GENERATION_DATASET_PATH,
        top_k: int = 5,
    ) -> None:
        self._pipeline = rag_pipeline
        self._panel = judge_panel
        self._dataset_path = dataset_path
        self._top_k = top_k

    def _load_dataset(self) -> GenerationGroundTruth:
        """Load the generation ground-truth dataset from JSON."""
        raw = self._dataset_path.read_text(encoding="utf-8")
        return GenerationGroundTruth.model_validate_json(raw)

    def run(self) -> GenerationEvaluationReport:
        """Execute the full generation evaluation.

        Returns:
            GenerationEvaluationReport with per-query and aggregate results.
        """
        dataset = self._load_dataset()
        logger.info(
            "Loaded generation dataset v%s with %d queries",
            dataset.version,
            len(dataset.annotations),
        )

        per_query_results: list[GenerationQueryResult] = []

        for i, annotation in enumerate(dataset.annotations):
            logger.info(
                "Evaluating query %d/%d: %s",
                i + 1,
                len(dataset.annotations),
                annotation.query[:80],
            )

            # Run RAG pipeline
            try:
                rag_response = self._pipeline.query(
                    annotation.query, top_k=self._top_k,
                )
            except Exception as e:
                logger.error("Pipeline error for query '%s': %s", annotation.query[:80], e)
                per_query_results.append(
                    GenerationQueryResult(
                        query=annotation.query,
                        generated_answer=f"[ERROR] {e}",
                        sources=[],
                        judge_scores=[],
                        latency_ms=0.0,
                        token_usage={},
                    )
                )
                continue

            # Extract context texts for judges
            context_texts = [
                src.chunk_summary for src in rag_response.sources
            ]

            # Run judges
            judge_scores = self._panel.evaluate_all(
                annotation.query,
                rag_response.answer,
                context_texts,
            )

            per_query_results.append(
                GenerationQueryResult(
                    query=annotation.query,
                    generated_answer=rag_response.answer,
                    sources=[s.source_path for s in rag_response.sources],
                    judge_scores=judge_scores,
                    latency_ms=rag_response.latency.total_ms,
                    token_usage=rag_response.token_usage.model_dump(),
                )
            )

        # Aggregate scores
        aggregate_scores, pass_rate = self._aggregate(per_query_results)

        return GenerationEvaluationReport(
            dataset_version=dataset.version,
            pipeline_config={"top_k": self._top_k},
            aggregate_scores=aggregate_scores,
            per_query_results=per_query_results,
            pass_rate=pass_rate,
        )

    @staticmethod
    def _aggregate(
        results: list[GenerationQueryResult],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute average score and pass rate per dimension."""
        if not results:
            return {}, {}

        dim_scores: dict[str, list[int]] = {d.value: [] for d in JudgeDimension}
        dim_passed: dict[str, list[bool]] = {d.value: [] for d in JudgeDimension}

        for qr in results:
            for js in qr.judge_scores:
                dim_scores[js.dimension.value].append(js.score)
                dim_passed[js.dimension.value].append(js.passed)

        aggregate_scores = {}
        pass_rate = {}
        for dim in JudgeDimension:
            scores = dim_scores[dim.value]
            passed = dim_passed[dim.value]
            if scores:
                aggregate_scores[dim.value] = round(sum(scores) / len(scores), 2)
                pass_rate[dim.value] = round(
                    sum(1 for p in passed if p) / len(passed) * 100, 1
                )
            else:
                aggregate_scores[dim.value] = 0.0
                pass_rate[dim.value] = 0.0

        return aggregate_scores, pass_rate

    @staticmethod
    def save_report(
        report: GenerationEvaluationReport,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> Path:
        """Save the report as a timestamped JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_generation_eval.json"
        filepath = output_dir / filename
        filepath.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Generation report saved to %s", filepath)
        return filepath

    @staticmethod
    def print_report(
        report: GenerationEvaluationReport, verbose: bool = False,
    ) -> None:
        """Print a formatted generation evaluation report to stdout."""
        print(f"\n{'=' * 60}")
        print("GENERATION EVALUATION REPORT")
        print(f"{'=' * 60}")
        print(f"Timestamp:       {report.timestamp.isoformat()}")
        print(f"Dataset version: {report.dataset_version}")
        print(f"Queries:         {len(report.per_query_results)}")

        print(f"\n{'─' * 40}")
        print("AGGREGATE SCORES (1-5)")
        print(f"{'─' * 40}")
        print(f"  {'Dimension':<20} {'Avg Score':>10} {'Pass Rate':>10}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 10}")
        for dim, score in report.aggregate_scores.items():
            rate = report.pass_rate.get(dim, 0.0)
            print(f"  {dim:<20} {score:>10.2f} {rate:>9.1f}%")

        if verbose and report.per_query_results:
            print(f"\n{'─' * 40}")
            print("PER-QUERY RESULTS")
            print(f"{'─' * 40}")
            for i, qr in enumerate(report.per_query_results, 1):
                print(f"\n  [{i}] {qr.query[:70]}")
                print(f"      Answer: {qr.generated_answer[:100]}...")
                print(f"      Sources: {len(qr.sources)}, Latency: {qr.latency_ms:.1f}ms")
                for js in qr.judge_scores:
                    status = "PASS" if js.passed else "FAIL"
                    print(
                        f"      {js.dimension.value}: {js.score}/5 [{status}] "
                        f"— {js.reasoning[:60]}"
                    )

        print(f"\n{'=' * 60}")
