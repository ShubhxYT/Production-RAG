"""Tests for LLM-based generation judges."""

import json
from unittest.mock import MagicMock

import pytest

from evaluation.generation_judges import GenerationJudge, JudgePanel
from evaluation.models import JudgeDimension, JudgeScore
from generation.models import GenerationConfig, GenerationResponse, TokenUsage


def _mock_provider(response_text: str) -> MagicMock:
    """Create a mock GenerationProvider that returns the given text."""
    provider = MagicMock()
    provider.generate.return_value = GenerationResponse(
        text=response_text,
        model="test-model",
        token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        finish_reason="stop",
    )
    return provider


class TestGenerationJudge:
    """Tests for a single GenerationJudge."""

    def test_evaluate_valid_json(self):
        response = json.dumps({"score": 4, "reasoning": "Well supported answer."})
        provider = _mock_provider(response)
        judge = GenerationJudge(provider, JudgeDimension.FAITHFULNESS)

        result = judge.evaluate("What is X?", "X is Y.", ["X is Y from source."])

        assert isinstance(result, JudgeScore)
        assert result.dimension == JudgeDimension.FAITHFULNESS
        assert result.score == 4
        assert result.reasoning == "Well supported answer."
        assert result.passed is True

    def test_evaluate_low_score_fails(self):
        response = json.dumps({"score": 2, "reasoning": "Fabricated claims."})
        provider = _mock_provider(response)
        judge = GenerationJudge(provider, JudgeDimension.FAITHFULNESS)

        result = judge.evaluate("What is X?", "X is Z.", ["X is Y."])

        assert result.score == 2
        assert result.passed is False

    def test_evaluate_fallback_regex_parse(self):
        # Malformed JSON but contains score
        response = 'The score is: score: 3. The answer is okay.'
        provider = _mock_provider(response)
        judge = GenerationJudge(provider, JudgeDimension.RELEVANCE)

        result = judge.evaluate("Q?", "A.", ["Context."])

        assert result.score == 3
        assert result.passed is True
        assert "(parsed from raw output)" in result.reasoning

    def test_evaluate_unparseable_returns_score_1(self):
        response = "I cannot evaluate this properly."
        provider = _mock_provider(response)
        judge = GenerationJudge(provider, JudgeDimension.COHERENCE)

        result = judge.evaluate("Q?", "A.", ["Context."])

        assert result.score == 1
        assert result.passed is False
        assert "(unparseable output)" in result.reasoning

    def test_score_clamped_to_valid_range(self):
        response = json.dumps({"score": 9, "reasoning": "Too high."})
        provider = _mock_provider(response)
        judge = GenerationJudge(provider, JudgeDimension.COMPLETENESS)

        result = judge.evaluate("Q?", "A.", ["C."])

        assert result.score == 5

    def test_each_dimension_works(self):
        response = json.dumps({"score": 4, "reasoning": "Good."})
        for dim in JudgeDimension:
            provider = _mock_provider(response)
            judge = GenerationJudge(provider, dim)
            result = judge.evaluate("Q?", "A.", ["C."])
            assert result.dimension == dim
            assert result.score == 4


class TestJudgePanel:
    """Tests for JudgePanel."""

    def test_evaluate_all_returns_all_dimensions(self):
        response = json.dumps({"score": 4, "reasoning": "Good."})
        provider = _mock_provider(response)
        config = GenerationConfig(temperature=0.1, max_output_tokens=256)

        judges = [
            GenerationJudge(provider, dim, config)
            for dim in JudgeDimension
        ]
        panel = JudgePanel(judges)

        scores = panel.evaluate_all("Q?", "A.", ["C."])

        assert len(scores) == 4
        dimensions = {s.dimension for s in scores}
        assert dimensions == {
            JudgeDimension.FAITHFULNESS,
            JudgeDimension.RELEVANCE,
            JudgeDimension.COMPLETENESS,
            JudgeDimension.COHERENCE,
        }

    def test_evaluate_all_with_mixed_scores(self):
        responses = [
            json.dumps({"score": 5, "reasoning": "Perfect."}),
            json.dumps({"score": 3, "reasoning": "Okay."}),
            json.dumps({"score": 1, "reasoning": "Bad."}),
            json.dumps({"score": 4, "reasoning": "Good."}),
        ]
        config = GenerationConfig(temperature=0.1, max_output_tokens=256)

        judges = []
        for dim, resp in zip(JudgeDimension, responses):
            provider = _mock_provider(resp)
            judges.append(GenerationJudge(provider, dim, config))

        panel = JudgePanel(judges)
        scores = panel.evaluate_all("Q?", "A.", ["C."])

        score_map = {s.dimension: s.score for s in scores}
        assert score_map[JudgeDimension.FAITHFULNESS] == 5
        assert score_map[JudgeDimension.RELEVANCE] == 3
        assert score_map[JudgeDimension.COMPLETENESS] == 1
        assert score_map[JudgeDimension.COHERENCE] == 4
