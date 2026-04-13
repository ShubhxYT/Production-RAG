"""Tests for the generation evaluation runner."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.generation_runner import GenerationEvaluationRunner
from evaluation.models import (
    GenerationEvaluationReport,
    JudgeDimension,
    JudgeScore,
)
from generation.models import TokenUsage
from pipeline.models import LatencyBreakdown, RAGResponse, SourceCitation


def _make_rag_response(answer: str = "Test answer.") -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sources=[
            SourceCitation(
                document_title="Test Doc",
                source_path="test/doc.md",
                chunk_summary="Test chunk summary.",
                page_numbers=[1],
                similarity_score=0.85,
            )
        ],
        latency=LatencyBreakdown(
            retrieval_ms=10.0, context_ms=1.0, generation_ms=50.0, total_ms=61.0,
        ),
        token_usage=TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
        ),
        prompt_version="qa_v1",
    )


def _make_judge_scores() -> list[JudgeScore]:
    return [
        JudgeScore(dimension=JudgeDimension.FAITHFULNESS, score=4, reasoning="Good.", passed=True),
        JudgeScore(dimension=JudgeDimension.RELEVANCE, score=5, reasoning="Great.", passed=True),
        JudgeScore(dimension=JudgeDimension.COMPLETENESS, score=3, reasoning="OK.", passed=True),
        JudgeScore(dimension=JudgeDimension.COHERENCE, score=4, reasoning="Clear.", passed=True),
    ]


def _make_dataset(num_queries: int = 2) -> dict:
    return {
        "version": "test-0.1",
        "created_at": "2026-04-04T00:00:00Z",
        "annotations": [
            {
                "query": f"Test query {i}?",
                "expected_answer": "",
                "relevant_chunk_ids": [],
                "tags": ["test"],
            }
            for i in range(num_queries)
        ],
    }


class TestGenerationEvaluationRunner:
    """Tests for GenerationEvaluationRunner."""

    def test_run_produces_report(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_rag_response()

        mock_panel = MagicMock()
        mock_panel.evaluate_all.return_value = _make_judge_scores()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(_make_dataset(2), f)
            dataset_path = Path(f.name)

        runner = GenerationEvaluationRunner(
            rag_pipeline=mock_pipeline,
            judge_panel=mock_panel,
            dataset_path=dataset_path,
        )
        report = runner.run()

        assert isinstance(report, GenerationEvaluationReport)
        assert len(report.per_query_results) == 2
        assert report.dataset_version == "test-0.1"

    def test_aggregate_scores_computed(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_rag_response()

        mock_panel = MagicMock()
        mock_panel.evaluate_all.return_value = _make_judge_scores()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(_make_dataset(2), f)
            dataset_path = Path(f.name)

        runner = GenerationEvaluationRunner(
            rag_pipeline=mock_pipeline,
            judge_panel=mock_panel,
            dataset_path=dataset_path,
        )
        report = runner.run()

        # Both queries get score 4 for faithfulness => avg = 4.0
        assert report.aggregate_scores["faithfulness"] == 4.0
        assert report.aggregate_scores["relevance"] == 5.0
        assert report.aggregate_scores["completeness"] == 3.0
        assert report.aggregate_scores["coherence"] == 4.0

    def test_pass_rate_computed(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_rag_response()

        mock_panel = MagicMock()
        mock_panel.evaluate_all.return_value = _make_judge_scores()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(_make_dataset(2), f)
            dataset_path = Path(f.name)

        runner = GenerationEvaluationRunner(
            rag_pipeline=mock_pipeline,
            judge_panel=mock_panel,
            dataset_path=dataset_path,
        )
        report = runner.run()

        # All scores >= 3, so pass rate should be 100%
        for dim in JudgeDimension:
            assert report.pass_rate[dim.value] == 100.0

    def test_empty_dataset_returns_empty_report(self):
        mock_pipeline = MagicMock()
        mock_panel = MagicMock()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(_make_dataset(0), f)
            dataset_path = Path(f.name)

        runner = GenerationEvaluationRunner(
            rag_pipeline=mock_pipeline,
            judge_panel=mock_panel,
            dataset_path=dataset_path,
        )
        report = runner.run()

        assert len(report.per_query_results) == 0
        assert report.aggregate_scores == {}

    def test_save_report_writes_json(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_rag_response()

        mock_panel = MagicMock()
        mock_panel.evaluate_all.return_value = _make_judge_scores()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(_make_dataset(1), f)
            dataset_path = Path(f.name)

        runner = GenerationEvaluationRunner(
            rag_pipeline=mock_pipeline,
            judge_panel=mock_panel,
            dataset_path=dataset_path,
        )
        report = runner.run()

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = GenerationEvaluationRunner.save_report(
                report, output_dir=Path(tmp_dir),
            )
            assert filepath.exists()
            data = json.loads(filepath.read_text())
            assert "aggregate_scores" in data
            assert "per_query_results" in data

    def test_print_report_runs_without_error(self, capsys):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = _make_rag_response()

        mock_panel = MagicMock()
        mock_panel.evaluate_all.return_value = _make_judge_scores()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(_make_dataset(1), f)
            dataset_path = Path(f.name)

        runner = GenerationEvaluationRunner(
            rag_pipeline=mock_pipeline,
            judge_panel=mock_panel,
            dataset_path=dataset_path,
        )
        report = runner.run()

        GenerationEvaluationRunner.print_report(report, verbose=True)

        captured = capsys.readouterr()
        assert "GENERATION EVALUATION REPORT" in captured.out
        assert "faithfulness" in captured.out

    def test_pipeline_error_handled_gracefully(self):
        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("LLM down")

        mock_panel = MagicMock()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(_make_dataset(1), f)
            dataset_path = Path(f.name)

        runner = GenerationEvaluationRunner(
            rag_pipeline=mock_pipeline,
            judge_panel=mock_panel,
            dataset_path=dataset_path,
        )
        report = runner.run()

        assert len(report.per_query_results) == 1
        assert "[ERROR]" in report.per_query_results[0].generated_answer
