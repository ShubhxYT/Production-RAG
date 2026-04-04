"""LLM-based judges for evaluating generation quality."""

import json
import logging
import re

from evaluation.models import JudgeDimension, JudgeScore
from generation.llm_service import GenerationProvider, get_generation_provider
from generation.models import GenerationConfig
from generation.prompt_templates import JUDGE_TEMPLATES

logger = logging.getLogger(__name__)


class GenerationJudge:
    """A single judge that evaluates one quality dimension.

    Uses an LLM to score answers on a specific dimension (faithfulness,
    relevance, completeness, or coherence).
    """

    def __init__(
        self,
        provider: GenerationProvider,
        dimension: JudgeDimension,
        config: GenerationConfig | None = None,
    ) -> None:
        self._provider = provider
        self._dimension = dimension
        self._config = config or GenerationConfig(
            temperature=0.1,
            max_output_tokens=256,
        )
        self._template = JUDGE_TEMPLATES[dimension.value]

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: list[str],
    ) -> JudgeScore:
        """Evaluate an answer on this judge's dimension.

        Args:
            query: The original user query.
            answer: The generated answer to evaluate.
            context_chunks: Context texts used for generation.

        Returns:
            JudgeScore with score, reasoning, and pass/fail.
        """
        system_prompt, user_prompt = self._template.render(
            query, answer, context_chunks,
        )

        response = self._provider.generate(
            system_prompt, user_prompt, self._config,
        )

        return self._parse_score(response.text)

    def _parse_score(self, raw_text: str) -> JudgeScore:
        """Parse LLM output into a JudgeScore.

        Tries JSON parsing first, falls back to regex extraction.
        """
        # Try JSON parse
        try:
            data = json.loads(raw_text.strip())
            score = int(data["score"])
            reasoning = str(data.get("reasoning", ""))
            score = max(1, min(5, score))
            return JudgeScore(
                dimension=self._dimension,
                score=score,
                reasoning=reasoning,
                passed=score >= 3,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        # Fallback: extract score with regex
        match = re.search(r'"?score"?\s*[:=]\s*(\d)', raw_text)
        if match:
            score = max(1, min(5, int(match.group(1))))
            return JudgeScore(
                dimension=self._dimension,
                score=score,
                reasoning=f"(parsed from raw output) {raw_text[:200]}",
                passed=score >= 3,
            )

        # Last resort: return lowest score
        logger.warning(
            "Could not parse judge output for %s: %s",
            self._dimension.value,
            raw_text[:200],
        )
        return JudgeScore(
            dimension=self._dimension,
            score=1,
            reasoning=f"(unparseable output) {raw_text[:200]}",
            passed=False,
        )


class JudgePanel:
    """Panel of judges that evaluate all quality dimensions.

    Runs each judge sequentially and collects scores for all dimensions.
    """

    def __init__(self, judges: list[GenerationJudge]) -> None:
        self._judges = judges

    def evaluate_all(
        self,
        query: str,
        answer: str,
        context_chunks: list[str],
    ) -> list[JudgeScore]:
        """Run all judges and return scores for each dimension.

        Args:
            query: The original user query.
            answer: The generated answer to evaluate.
            context_chunks: Context texts used for generation.

        Returns:
            List of JudgeScore, one per dimension.
        """
        scores: list[JudgeScore] = []
        for judge in self._judges:
            logger.info("Running %s judge...", judge._dimension.value)
            score = judge.evaluate(query, answer, context_chunks)
            logger.info(
                "%s: score=%d, passed=%s",
                score.dimension.value,
                score.score,
                score.passed,
            )
            scores.append(score)
        return scores

    @classmethod
    def default_panel(
        cls,
        provider_name: str = "gemini",
    ) -> "JudgePanel":
        """Create a panel with all four judge dimensions.

        Args:
            provider_name: LLM provider to use ('gemini' or 'cerebras').

        Returns:
            JudgePanel with faithfulness, relevance, completeness, coherence judges.
        """
        provider = get_generation_provider(provider_name)
        config = GenerationConfig(temperature=0.1, max_output_tokens=256)
        judges = [
            GenerationJudge(provider, dim, config)
            for dim in JudgeDimension
        ]
        return cls(judges)
