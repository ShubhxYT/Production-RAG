"""Tests for continuous evaluation and scheduler."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.continuous import ContinuousEvaluator
from evaluation.models import EvaluationReport, MetricResult


def _make_report(precision_at_5: float) -> EvaluationReport:
    """Create a minimal EvaluationReport with a given precision@5."""
    return EvaluationReport(
        timestamp=datetime.now(timezone.utc),
        dataset_version="1.0.0",
        retrieval_config={"top_k": 5},
        aggregate_metrics=[
            MetricResult(metric_name="precision@5", value=precision_at_5, k=5),
        ],
        per_query_results=[],
    )


class TestContinuousEvaluator:
    """Tests for ContinuousEvaluator."""

    def test_run_calls_save_report(self):
        """ContinuousEvaluator.run() saves the evaluation report."""
        with patch("retrieval.service.RetrievalService") as MockService, \
             patch("evaluation.retrieval_runner.EvaluationRunner") as MockRunner:
            
            mock_runner = MockRunner.return_value
            report = _make_report(0.80)
            mock_runner.run.return_value = report

            tmp_path = Path("/tmp/test_eval_report.json")
            MockRunner.save_report.return_value = tmp_path

            evaluator = ContinuousEvaluator()

            with patch.object(evaluator, "_load_previous_report", return_value=None):
                evaluator.run()

            MockRunner.save_report.assert_called_once_with(report)

    def test_run_skips_when_no_dataset(self, tmp_path):
        """ContinuousEvaluator.run() skips when dataset file is missing."""
        evaluator = ContinuousEvaluator()

        with patch("evaluation.continuous.DEFAULT_DATASET_PATH", tmp_path / "nonexistent.json"):
            evaluator.run()


class TestCheckRegression:
    """Tests for ContinuousEvaluator._check_regression."""

    def test_regression_detected(self, caplog):
        """A >10% drop in precision@5 triggers a WARNING."""
        evaluator = ContinuousEvaluator()
        current = _make_report(0.65)
        previous = _make_report(0.80)

        with caplog.at_level(logging.WARNING):
            evaluator._check_regression(current, previous)

        assert any("REGRESSION DETECTED" in msg for msg in caplog.messages)

    def test_no_regression_small_drop(self, caplog):
        """A <10% drop in precision@5 does NOT trigger a WARNING."""
        evaluator = ContinuousEvaluator()
        current = _make_report(0.78)
        previous = _make_report(0.80)

        with caplog.at_level(logging.WARNING):
            evaluator._check_regression(current, previous)

        assert not any("REGRESSION DETECTED" in msg for msg in caplog.messages)

    def test_no_regression_improvement(self, caplog):
        """An improvement does NOT trigger a WARNING."""
        evaluator = ContinuousEvaluator()
        current = _make_report(0.90)
        previous = _make_report(0.80)

        with caplog.at_level(logging.WARNING):
            evaluator._check_regression(current, previous)

        assert not any("REGRESSION DETECTED" in msg for msg in caplog.messages)


class TestBackgroundEvaluationScheduler:
    """Tests for BackgroundEvaluationScheduler."""

    @patch("apscheduler.triggers.interval.IntervalTrigger")
    @patch("apscheduler.schedulers.background.BackgroundScheduler")
    def test_start_and_stop(self, MockBGScheduler, MockTrigger):
        """Scheduler starts with correct interval and shuts down cleanly."""
        from evaluation.scheduler import BackgroundEvaluationScheduler

        mock_scheduler = MockBGScheduler.return_value
        mock_scheduler.running = True

        scheduler = BackgroundEvaluationScheduler()
        scheduler.start(interval_hours=24)

        MockBGScheduler.assert_called_once()
        mock_scheduler.add_job.assert_called_once()
        MockTrigger.assert_called_once_with(hours=24)
        mock_scheduler.start.assert_called_once()

        scheduler.stop()
        mock_scheduler.shutdown.assert_called_once_with(wait=False)

    def test_stop_without_start(self):
        """Stopping without starting does not raise."""
        from evaluation.scheduler import BackgroundEvaluationScheduler

        scheduler = BackgroundEvaluationScheduler()
        scheduler.stop()  # Should not raise
