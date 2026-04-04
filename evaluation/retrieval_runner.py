"""Evaluation runner that orchestrates retrieval evaluation against a ground-truth dataset."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from evaluation.models import (
    EvaluationReport,
    GroundTruthDataset,
    MetricResult,
    QueryResult,
)
from evaluation.retrieval_metrics import compute_all_metrics
from retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("evaluation/datasets/retrieval_ground_truth.json")
DEFAULT_OUTPUT_DIR = Path("evaluation/results")
DEFAULT_K_VALUES = [1, 3, 5, 10]


class EvaluationRunner:
    """Runs retrieval evaluation against a ground-truth dataset.

    For each annotated query, retrieves top-k chunks using the
    RetrievalService, computes metrics against known relevant chunk IDs,
    and aggregates into a report.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        top_k: int = 10,
        threshold: float | None = None,
        k_values: list[int] | None = None,
    ) -> None:
        self._service = retrieval_service
        self._dataset_path = dataset_path
        self._top_k = top_k
        self._threshold = threshold
        self._k_values = k_values or DEFAULT_K_VALUES

    def _load_dataset(self) -> GroundTruthDataset:
        """Load and validate the ground-truth dataset from JSON."""
        raw = self._dataset_path.read_text(encoding="utf-8")
        return GroundTruthDataset.model_validate_json(raw)

    def run(self) -> EvaluationReport:
        """Execute the full evaluation and return a report.

        Returns:
            EvaluationReport with aggregate and per-query metrics.
        """
        dataset = self._load_dataset()
        logger.info(
            "Loaded ground-truth dataset v%s with %d annotations",
            dataset.version,
            len(dataset.annotations),
        )

        per_query_results: list[QueryResult] = []

        for i, annotation in enumerate(dataset.annotations):
            logger.info(
                "Evaluating query %d/%d: %s",
                i + 1,
                len(dataset.annotations),
                annotation.query[:80],
            )

            # Retrieve using the configured top_k (retrieve enough for max k_value)
            max_k = max(self._k_values) if self._k_values else self._top_k
            retrieve_k = max(max_k, self._top_k)
            response = self._service.retrieve(
                query=annotation.query,
                top_k=retrieve_k,
                threshold=self._threshold,
            )

            retrieved_ids = [r.chunk_id for r in response.results]
            relevant_ids = set(annotation.relevant_chunk_ids)

            # Compute metrics at each k value
            query_metrics: list[MetricResult] = []
            for k in self._k_values:
                query_metrics.extend(compute_all_metrics(retrieved_ids, relevant_ids, k))

            per_query_results.append(
                QueryResult(
                    query=annotation.query,
                    tags=annotation.tags,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=annotation.relevant_chunk_ids,
                    metrics=query_metrics,
                )
            )

        # Aggregate metrics: mean of each metric across all queries
        aggregate = self._aggregate_metrics(per_query_results)

        return EvaluationReport(
            dataset_version=dataset.version,
            retrieval_config={
                "top_k": self._top_k,
                "threshold": self._threshold,
                "k_values": self._k_values,
            },
            aggregate_metrics=aggregate,
            per_query_results=per_query_results,
        )

    def _aggregate_metrics(
        self, per_query: list[QueryResult]
    ) -> list[MetricResult]:
        """Average each metric across all queries."""
        if not per_query:
            return []

        # Collect all metric names across all queries
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        metric_k: dict[str, int | None] = {}

        for qr in per_query:
            for m in qr.metrics:
                metric_sums[m.metric_name] = metric_sums.get(m.metric_name, 0.0) + m.value
                metric_counts[m.metric_name] = metric_counts.get(m.metric_name, 0) + 1
                metric_k[m.metric_name] = m.k

        return [
            MetricResult(
                metric_name=name,
                value=round(metric_sums[name] / metric_counts[name], 4),
                k=metric_k[name],
            )
            for name in sorted(metric_sums.keys())
        ]

    @staticmethod
    def save_report(report: EvaluationReport, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
        """Save the report as a timestamped JSON file.

        Args:
            report: The evaluation report to save.
            output_dir: Directory to write the report to.

        Returns:
            Path to the saved report file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_retrieval_eval.json"
        filepath = output_dir / filename

        filepath.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Report saved to %s", filepath)
        return filepath

    @staticmethod
    def print_report(report: EvaluationReport, verbose: bool = False) -> None:
        """Print a formatted evaluation report to stdout.

        Args:
            report: The evaluation report to print.
            verbose: If True, also print per-query results.
        """
        print(f"\n{'=' * 60}")
        print("RETRIEVAL EVALUATION REPORT")
        print(f"{'=' * 60}")
        print(f"Timestamp:       {report.timestamp.isoformat()}")
        print(f"Dataset version: {report.dataset_version}")
        print(f"Config:          {report.retrieval_config}")
        print(f"Queries:         {len(report.per_query_results)}")

        print(f"\n{'─' * 40}")
        print("AGGREGATE METRICS")
        print(f"{'─' * 40}")
        print(f"  {'Metric':<20} {'Value':>10}")
        print(f"  {'─' * 20} {'─' * 10}")
        for m in report.aggregate_metrics:
            print(f"  {m.metric_name:<20} {m.value:>10.4f}")

        if verbose and report.per_query_results:
            print(f"\n{'─' * 40}")
            print("PER-QUERY RESULTS")
            print(f"{'─' * 40}")
            for i, qr in enumerate(report.per_query_results, 1):
                print(f"\n  [{i}] {qr.query[:70]}")
                print(f"      Tags: {qr.tags}")
                print(f"      Retrieved: {len(qr.retrieved_ids)} chunks")
                print(f"      Relevant:  {len(qr.relevant_ids)} chunks")
                for m in qr.metrics:
                    print(f"      {m.metric_name}: {m.value:.4f}")

        print(f"\n{'=' * 60}")
