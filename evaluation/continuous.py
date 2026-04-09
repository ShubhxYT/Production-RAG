"""Continuous evaluator that runs retrieval evaluation and detects regressions."""

from pathlib import Path

from observability.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = Path("evaluation/datasets/retrieval_ground_truth.json")
DEFAULT_RESULTS_DIR = Path("evaluation/results")


class ContinuousEvaluator:
    """Runs scheduled retrieval evaluation and checks for regressions."""

    def run(self, top_k: int = 5) -> None:
        """Execute a retrieval evaluation run and check for regressions.

        Args:
            top_k: Number of results to retrieve per query.
        """
        dataset_path = DEFAULT_DATASET_PATH
        if not dataset_path.exists():
            logger.warning(
                "Ground-truth dataset not found at %s, skipping evaluation",
                dataset_path,
            )
            return

        from evaluation.retrieval_runner import EvaluationRunner
        from retrieval.service import RetrievalService

        service = RetrievalService()
        runner = EvaluationRunner(
            retrieval_service=service,
            dataset_path=dataset_path,
            top_k=top_k,
        )

        report = runner.run()
        saved_path = EvaluationRunner.save_report(report)

        prev_report = self._load_previous_report(saved_path)
        if prev_report is not None:
            self._check_regression(report, prev_report)

        # Extract precision@5 for logging
        p5 = None
        for m in report.aggregate_metrics:
            if m.metric_name == "precision@5":
                p5 = m.value
                break

        logger.info(
            "Scheduled evaluation complete",
            extra={"precision@5": p5, "saved_to": str(saved_path)},
        )

    def _load_previous_report(self, current_path: Path):
        """Load the most recent evaluation report before the current one.

        Args:
            current_path: Path to the just-saved current report.

        Returns:
            EvaluationReport or None if no prior reports exist.
        """
        from evaluation.models import EvaluationReport

        results_dir = DEFAULT_RESULTS_DIR
        if not results_dir.exists():
            return None

        candidates = sorted(results_dir.glob("*_retrieval_eval.json"))
        # Exclude the current report
        candidates = [p for p in candidates if p.resolve() != current_path.resolve()]

        if not candidates:
            return None

        latest = candidates[-1]
        try:
            raw = latest.read_text(encoding="utf-8")
            return EvaluationReport.model_validate_json(raw)
        except Exception as e:
            logger.warning("Failed to load previous report %s: %s", latest, e)
            return None

    def _check_regression(self, current, previous) -> None:
        """Compare precision@5 between current and previous reports.

        Logs a WARNING if precision@5 drops more than 10%.

        Args:
            current: The current EvaluationReport.
            previous: The previous EvaluationReport.
        """
        current_p5 = None
        previous_p5 = None

        for m in current.aggregate_metrics:
            if m.metric_name == "precision@5":
                current_p5 = m.value
                break

        for m in previous.aggregate_metrics:
            if m.metric_name == "precision@5":
                previous_p5 = m.value
                break

        if current_p5 is None or previous_p5 is None:
            return

        if previous_p5 == 0:
            return

        delta = (current_p5 - previous_p5) / previous_p5

        if delta < -0.10:
            drop_pct = abs(delta) * 100
            logger.warning(
                "REGRESSION DETECTED: precision@5 dropped %.1f%% (%.4f -> %.4f)",
                drop_pct,
                previous_p5,
                current_p5,
                extra={
                    "previous_p5": previous_p5,
                    "current_p5": current_p5,
                    "drop_pct": round(drop_pct, 1),
                },
            )
