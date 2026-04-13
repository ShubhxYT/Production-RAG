"""Tests for retrieval evaluation metrics.

Run: python -m pytest test/test_retrieval_metrics.py -v
"""

import math

import pytest

from evaluation.models import MetricResult
from evaluation.retrieval_metrics import (
    compute_all_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_retrieval(self):
        """All top-k results are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_no_relevant_results(self):
        """None of the top-k results are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)

    def test_partial_relevant(self):
        """2 of 5 top results are relevant."""
        retrieved = ["a", "x", "b", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)

    def test_k_smaller_than_retrieved(self):
        """Only look at first k results."""
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        # At k=2: only ["a", "x"] -> 1/2
        assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    def test_k_larger_than_retrieved(self):
        """k exceeds the number of retrieved results."""
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        # Only 2 retrieved, both relevant -> 2/2
        assert precision_at_k(retrieved, relevant, k=10) == pytest.approx(1.0)

    def test_empty_retrieved(self):
        retrieved: list[str] = []
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.0)

    def test_k_zero(self):
        retrieved = ["a"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self):
        """All relevant items found in top-k."""
        retrieved = ["a", "b", "c", "x"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(1.0)

    def test_no_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)

    def test_partial_recall(self):
        """1 of 3 relevant items found."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_k_limits_recall(self):
        """Relevant items exist but beyond k."""
        retrieved = ["x", "y", "a", "b"]
        relevant = {"a", "b"}
        # At k=2: only ["x", "y"] -> 0/2 relevant found
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)

    def test_empty_relevant_set(self):
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        retrieved: list[str] = []
        relevant = {"a"}
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.0)

    def test_k_zero(self):
        retrieved = ["a"]
        relevant = {"a"}
        assert recall_at_k(retrieved, relevant, k=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_result_relevant(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.0)

    def test_multiple_relevant_returns_first(self):
        """MRR uses the rank of the *first* relevant result."""
        retrieved = ["x", "a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_empty_retrieved(self):
        retrieved: list[str] = []
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self):
        """All relevant items ranked at the top -> NDCG = 1.0."""
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """No relevant items retrieved -> NDCG = 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)

    def test_single_relevant_at_position_1(self):
        """One relevant item at rank 1 out of 1 relevant total."""
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        # DCG = 1/log2(2) = 1.0; IDCG = 1/log2(2) = 1.0
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_single_relevant_at_position_3(self):
        """One relevant item at rank 3."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        # DCG = 1/log2(4); IDCG = 1/log2(2)
        expected = (1 / math.log2(4)) / (1 / math.log2(2))
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)

    def test_partial_relevant_ordering_matters(self):
        """Two relevant items; order affects NDCG."""
        relevant = {"a", "b"}

        # Good order: relevant at positions 1, 2
        good_order = ["a", "b", "x", "y", "z"]
        ndcg_good = ndcg_at_k(good_order, relevant, k=5)

        # Bad order: relevant at positions 3, 5
        bad_order = ["x", "y", "a", "z", "b"]
        ndcg_bad = ndcg_at_k(bad_order, relevant, k=5)

        assert ndcg_good > ndcg_bad
        assert ndcg_good == pytest.approx(1.0)

    def test_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        retrieved: list[str] = []
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(0.0)

    def test_k_zero(self):
        retrieved = ["a"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_returns_all_four_metrics(self):
        retrieved = ["a", "b", "x"]
        relevant = {"a", "b"}
        results = compute_all_metrics(retrieved, relevant, k=3)

        assert len(results) == 4
        names = {r.metric_name for r in results}
        assert names == {"precision@3", "recall@3", "mrr", "ndcg@3"}

    def test_metric_values_consistent(self):
        """Cross-check individual functions match aggregate output."""
        retrieved = ["x", "a", "y", "b", "z"]
        relevant = {"a", "b", "c"}
        k = 5

        results = compute_all_metrics(retrieved, relevant, k)
        by_name = {r.metric_name: r.value for r in results}

        assert by_name[f"precision@{k}"] == pytest.approx(
            precision_at_k(retrieved, relevant, k)
        )
        assert by_name[f"recall@{k}"] == pytest.approx(
            recall_at_k(retrieved, relevant, k)
        )
        assert by_name["mrr"] == pytest.approx(
            mean_reciprocal_rank(retrieved, relevant)
        )
        assert by_name[f"ndcg@{k}"] == pytest.approx(
            ndcg_at_k(retrieved, relevant, k)
        )

    def test_all_metrics_are_metric_result_instances(self):
        results = compute_all_metrics(["a"], {"a"}, k=1)
        for r in results:
            assert isinstance(r, MetricResult)

    def test_k_value_stored(self):
        results = compute_all_metrics(["a"], {"a"}, k=7)
        by_name = {r.metric_name: r for r in results}
        assert by_name["precision@7"].k == 7
        assert by_name["recall@7"].k == 7
        assert by_name["mrr"].k is None
        assert by_name["ndcg@7"].k == 7
