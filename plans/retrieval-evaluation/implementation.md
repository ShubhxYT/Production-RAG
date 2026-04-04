# Retrieval Evaluation (Precision/Recall)

**Branch:** `feat/retrieval-evaluation`
**Description:** Build a retrieval evaluation framework with ground-truth datasets, standard IR metrics (Precision@k, Recall@k, MRR, NDCG@k), and an automated evaluation runner that establishes a baseline for retrieval quality.

## Goal
Create a repeatable evaluation framework that measures retrieval quality using standard IR metrics against a ground-truth dataset, runnable from CLI and importable for automated pipelines.

## Prerequisites
Make sure that the user is currently on the `feat/retrieval-evaluation` branch before beginning implementation.
If not, move them to the correct branch. If the branch does not exist, create it from main.

## Implementation Steps

### Step 1: Create evaluation module with Pydantic models and seed dataset

- [x] Create the `evaluation/` directory
- [x] Create `evaluation/__init__.py`:

```python
"""Evaluation framework for measuring retrieval and generation quality."""

from evaluation.models import (
    EvaluationReport,
    GroundTruthDataset,
    MetricResult,
    RelevanceAnnotation,
)

__all__ = [
    "EvaluationReport",
    "GroundTruthDataset",
    "MetricResult",
    "RelevanceAnnotation",
]
```

- [x] Create `evaluation/models.py`:

```python
"""Pydantic models for evaluation datasets, metrics, and reports."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class RelevanceAnnotation(BaseModel):
    """A single query with its ground-truth relevant chunk IDs."""

    query: str = Field(description="The evaluation query.")
    relevant_chunk_ids: list[str] = Field(
        description="UUIDs of chunks that are relevant to this query."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Query category tags (e.g. 'factual', 'paraphrased', 'multi-hop').",
    )
    notes: str = Field(
        default="",
        description="Optional notes about why these chunks are relevant.",
    )


class GroundTruthDataset(BaseModel):
    """A versioned collection of query-relevance annotations."""

    version: str = Field(description="Semantic version of this dataset.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this dataset was created.",
    )
    annotations: list[RelevanceAnnotation] = Field(
        default_factory=list,
        description="List of query-relevance annotations.",
    )


class MetricResult(BaseModel):
    """A single evaluation metric value."""

    metric_name: str = Field(description="Name of the metric (e.g. 'precision@5').")
    value: float = Field(description="Metric value.")
    k: int | None = Field(default=None, description="The k value used, if applicable.")


class QueryResult(BaseModel):
    """Per-query evaluation results."""

    query: str = Field(description="The query text.")
    tags: list[str] = Field(default_factory=list, description="Query tags.")
    retrieved_ids: list[str] = Field(
        default_factory=list, description="Chunk IDs returned by retrieval."
    )
    relevant_ids: list[str] = Field(
        default_factory=list, description="Ground-truth relevant chunk IDs."
    )
    metrics: list[MetricResult] = Field(
        default_factory=list, description="Metrics computed for this query."
    )


class EvaluationReport(BaseModel):
    """Full evaluation report with aggregate and per-query results."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evaluation was run.",
    )
    dataset_version: str = Field(description="Version of the ground-truth dataset used.")
    retrieval_config: dict = Field(
        default_factory=dict,
        description="Retrieval parameters (top_k, threshold, model, etc.).",
    )
    aggregate_metrics: list[MetricResult] = Field(
        default_factory=list,
        description="Metrics averaged across all queries.",
    )
    per_query_results: list[QueryResult] = Field(
        default_factory=list,
        description="Per-query retrieval results and metrics.",
    )
```

- [x] Create the `evaluation/datasets/` directory
- [x] Create `evaluation/datasets/retrieval_ground_truth.json`:

> **Note:** This is a skeleton seed dataset. After running, inspect the database to
> find real chunk IDs and replace the placeholders. Use `hypothetical_questions`
> stored on chunks as a source for realistic queries, and map each question back
> to its owning chunk ID.

```json
{
  "version": "0.1.0",
  "created_at": "2026-04-04T00:00:00Z",
  "annotations": [
    {
      "query": "What is the glass transition temperature of polymers?",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_1"],
      "tags": ["factual"],
      "notes": "Populate from Polymers-Lecture chunks."
    },
    {
      "query": "Explain the cross-linking process in polymer chemistry",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_2"],
      "tags": ["factual"],
      "notes": "Populate from Polymers-Lecture chunks."
    },
    {
      "query": "What datasets are available for the Time Waster Retreat model?",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_3"],
      "tags": ["factual"],
      "notes": "Populate from READ ME Time Waster Retreat chunks."
    },
    {
      "query": "What are the benefits of Sampoorna Raksha Promise plan?",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_4"],
      "tags": ["factual"],
      "notes": "Populate from Sampoorna-Raksha-Promise chunks."
    },
    {
      "query": "Tell me about Life Bricks use cases",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_5"],
      "tags": ["factual"],
      "notes": "Populate from Life Bricks chunks."
    },
    {
      "query": "What programming skills does Shubh have?",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_6"],
      "tags": ["factual"],
      "notes": "Populate from CV-Shubh chunks."
    },
    {
      "query": "polymer molecular weight distribution",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_7"],
      "tags": ["paraphrased"],
      "notes": "Different wording for polymer lecture content."
    },
    {
      "query": "insurance plan coverage details",
      "relevant_chunk_ids": ["REPLACE_WITH_REAL_CHUNK_ID_8"],
      "tags": ["paraphrased"],
      "notes": "Paraphrased query for Sampoorna Raksha content."
    },
    {
      "query": "What are the key polymer properties discussed in lectures 1-3?",
      "relevant_chunk_ids": [
        "REPLACE_WITH_REAL_CHUNK_ID_9",
        "REPLACE_WITH_REAL_CHUNK_ID_10"
      ],
      "tags": ["multi-hop"],
      "notes": "Requires chunks from multiple sections."
    },
    {
      "query": "completely unrelated topic about quantum computing",
      "relevant_chunk_ids": [],
      "tags": ["negative"],
      "notes": "No relevant docs expected — tests low-score behavior."
    }
  ]
}
```

##### Step 1 Verification Checklist
- [x] No import errors: `python -c "from evaluation.models import RelevanceAnnotation, GroundTruthDataset, MetricResult, QueryResult, EvaluationReport; print('OK')"`
- [x] Models serialize correctly: `python -c "from evaluation.models import RelevanceAnnotation; r = RelevanceAnnotation(query='test', relevant_chunk_ids=['abc']); print(r.model_dump_json(indent=2))"`
- [x] Seed dataset loads: `python -c "import json; from pathlib import Path; from evaluation.models import GroundTruthDataset; d = GroundTruthDataset.model_validate_json(Path('evaluation/datasets/retrieval_ground_truth.json').read_text()); print(f'{len(d.annotations)} annotations loaded')"`
- [x] Module imports work: `python -c "from evaluation import EvaluationReport, GroundTruthDataset, MetricResult, RelevanceAnnotation; print('OK')"`

#### Step 1 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.
```
git add evaluation/ && git commit -m "feat(evaluation): add evaluation models and seed ground-truth dataset"
```

---

### Step 2: Implement retrieval metric computation functions

- [x] Create `evaluation/retrieval_metrics.py`:

```python
"""Information retrieval metrics: Precision@k, Recall@k, MRR, NDCG@k."""

import math

from evaluation.models import MetricResult


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of the top-k retrieved results that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Number of top results to consider.

    Returns:
        Precision@k as a float in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for rid in top_k if rid in relevant_ids)
    return relevant_in_top_k / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of all relevant items found in the top-k retrieved results.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Number of top results to consider.

    Returns:
        Recall@k as a float in [0, 1]. Returns 0.0 if there are no relevant items.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for rid in top_k if rid in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Reciprocal of the rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_ids: Set of ground-truth relevant chunk IDs.

    Returns:
        1/rank of first relevant result, or 0.0 if none found.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance).

    Uses binary relevance: 1 if chunk is relevant, 0 otherwise.
    DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = sum_{i=1}^{min(k, |relevant|)} 1 / log2(i + 1)

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Number of top results to consider.

    Returns:
        NDCG@k as a float in [0, 1]. Returns 0.0 if there are no relevant items.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, rid in enumerate(top_k):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-indexed, rank is i+1

    # IDCG: perfect ranking — all relevant items at the top
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_all_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> list[MetricResult]:
    """Compute all retrieval metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (best first).
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Number of top results to consider.

    Returns:
        List of MetricResult for precision@k, recall@k, MRR, and NDCG@k.
    """
    return [
        MetricResult(
            metric_name=f"precision@{k}",
            value=precision_at_k(retrieved_ids, relevant_ids, k),
            k=k,
        ),
        MetricResult(
            metric_name=f"recall@{k}",
            value=recall_at_k(retrieved_ids, relevant_ids, k),
            k=k,
        ),
        MetricResult(
            metric_name="mrr",
            value=mean_reciprocal_rank(retrieved_ids, relevant_ids),
            k=None,
        ),
        MetricResult(
            metric_name=f"ndcg@{k}",
            value=ndcg_at_k(retrieved_ids, relevant_ids, k),
            k=k,
        ),
    ]
```

- [x] Create `test/test_retrieval_metrics.py`:

```python
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
```

##### Step 2 Verification Checklist
- [x] No import errors: `python -c "from evaluation.retrieval_metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k, compute_all_metrics; print('OK')"`
- [x] All tests pass: `python -m pytest test/test_retrieval_metrics.py -v`
- [x] Spot-check a hand-computed example: `python -c "from evaluation.retrieval_metrics import precision_at_k; print(precision_at_k(['a','x','b'], {'a','b'}, k=3))"`  — should print `0.6666...`

#### Step 2 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.
```
git add evaluation/retrieval_metrics.py test/test_retrieval_metrics.py && git commit -m "feat(evaluation): implement retrieval metric functions with tests"
```

---

### Step 3: Implement evaluation runner and CLI

- [ ] Create `evaluation/retrieval_runner.py`:

```python
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
```

- [ ] Create `evaluation/cli.py`:

```python
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
```

- [ ] Create `evaluation/__main__.py`:

```python
"""Allow running the evaluation module with python -m evaluation."""

from evaluation.cli import main

main()
```

- [ ] Create `test/test_evaluation_runner.py`:

```python
"""Tests for the evaluation runner.

Requires a running PostgreSQL instance with pgvector.
Run: docker compose -f pgvector.yaml up -d
Then: alembic upgrade head
Then: python -m pytest test/test_evaluation_runner.py -v
"""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from database.connection import get_session
from database.repository import DocumentRepository
from embeddings.models import EmbeddingConfig, EmbeddingResult
from embeddings.service import EmbeddingService
from evaluation.models import EvaluationReport, GroundTruthDataset
from evaluation.retrieval_runner import EvaluationRunner
from ingestion.models import Chunk, Document, ElementType
from retrieval.service import RetrievalService


# ---------------------------------------------------------------------------
# Helpers (reused pattern from test_retrieval.py)
# ---------------------------------------------------------------------------


def _make_document(
    source_path: str | None = None,
    title: str = "Eval Test Document",
    num_chunks: int = 3,
) -> Document:
    """Create a test Document with chunks."""
    doc_id = str(uuid.uuid4())
    source_path = source_path or f"test/{doc_id}.md"
    chunks = []
    for i in range(num_chunks):
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=f"Test chunk {i} about evaluation metrics. " * 10,
                token_count=50,
                document_id=doc_id,
                section_path=["Section 1", f"Subsection {i}"],
                page_numbers=[i + 1],
                element_types=[ElementType.PARAGRAPH],
                position=i,
                overlap_before="" if i == 0 else f"overlap from chunk {i - 1}",
                summary=f"Summary of evaluation chunk {i}.",
                keywords=["evaluation", f"chunk{i}"],
                hypothetical_questions=[f"What is evaluation chunk {i}?"],
            )
        )
    return Document(
        id=doc_id,
        source_path=source_path,
        title=title,
        format="md",
        raw_content="Full raw content here.",
        chunks=chunks,
        created_at=datetime.now(timezone.utc),
    )


def _make_embedding(dim: int = 768) -> list[float]:
    """Create a random-ish embedding vector."""
    import random

    random.seed(42)
    return [random.uniform(-1, 1) for _ in range(dim)]


class MockProvider:
    """Mock embedding provider for testing."""

    def __init__(self, fixed_vector: list[float] | None = None, dim: int = 768) -> None:
        self.dim = dim
        self.fixed_vector = fixed_vector

    def embed(self, texts: list[str], config: EmbeddingConfig) -> EmbeddingResult:
        if self.fixed_vector:
            vectors = [self.fixed_vector for _ in texts]
        else:
            vectors = [[0.1] * self.dim for _ in texts]
        return EmbeddingResult(
            vectors=vectors,
            model=config.model_name,
            dimensions=self.dim,
            token_usage=0,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session():
    """Provide a database session that rolls back after each test."""
    sess = get_session()
    yield sess
    sess.rollback()
    sess.close()


@pytest.fixture()
def repo():
    return DocumentRepository()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestGroundTruthDataset:
    def test_load_from_json(self):
        """Verify GroundTruthDataset can be loaded from a JSON string."""
        data = {
            "version": "0.1.0",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [
                {
                    "query": "test query",
                    "relevant_chunk_ids": ["id1", "id2"],
                    "tags": ["factual"],
                    "notes": "test note",
                }
            ],
        }
        ds = GroundTruthDataset.model_validate(data)
        assert ds.version == "0.1.0"
        assert len(ds.annotations) == 1
        assert ds.annotations[0].query == "test query"
        assert len(ds.annotations[0].relevant_chunk_ids) == 2

    def test_roundtrip_json(self):
        data = {
            "version": "1.0.0",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [],
        }
        ds = GroundTruthDataset.model_validate(data)
        json_str = ds.model_dump_json()
        restored = GroundTruthDataset.model_validate_json(json_str)
        assert restored.version == "1.0.0"


class TestEvaluationReport:
    def test_report_roundtrip(self):
        report = EvaluationReport(
            dataset_version="0.1.0",
            retrieval_config={"top_k": 5},
            aggregate_metrics=[],
            per_query_results=[],
        )
        json_str = report.model_dump_json()
        restored = EvaluationReport.model_validate_json(json_str)
        assert restored.dataset_version == "0.1.0"


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


class TestEvaluationRunner:
    def _setup(self, session, repo):
        """Insert test data and return (service, chunk_ids)."""
        doc = _make_document()
        repo.insert_document(session, doc)

        vec = _make_embedding()
        repo.insert_bulk_embeddings(
            session,
            [(c.id, vec) for c in doc.chunks],
            model_name="BAAI/bge-base-en-v1.5",
        )

        provider = MockProvider(fixed_vector=vec)
        config = EmbeddingConfig()
        embed_service = EmbeddingService(provider=provider, config=config)
        service = RetrievalService(embedding_service=embed_service, config=config)

        chunk_ids = [c.id for c in doc.chunks]
        return service, chunk_ids

    def test_run_produces_report(self, session, repo):
        """Runner produces a valid EvaluationReport."""
        service, chunk_ids = self._setup(session, repo)

        # Create a temp ground-truth dataset referencing actual chunk IDs
        dataset = {
            "version": "test",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [
                {
                    "query": "evaluation metrics",
                    "relevant_chunk_ids": [chunk_ids[0]],
                    "tags": ["factual"],
                },
                {
                    "query": "no match query",
                    "relevant_chunk_ids": ["nonexistent-id"],
                    "tags": ["negative"],
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dataset, f)
            dataset_path = Path(f.name)

        runner = EvaluationRunner(
            retrieval_service=service,
            dataset_path=dataset_path,
            top_k=5,
            k_values=[1, 3, 5],
        )

        with patch("retrieval.service.get_session", return_value=session):
            report = runner.run()

        assert isinstance(report, EvaluationReport)
        assert report.dataset_version == "test"
        assert len(report.per_query_results) == 2
        assert len(report.aggregate_metrics) > 0

        # Metrics should include precision, recall, mrr, ndcg for each k
        metric_names = {m.metric_name for m in report.aggregate_metrics}
        assert "precision@5" in metric_names
        assert "recall@5" in metric_names
        assert "mrr" in metric_names
        assert "ndcg@5" in metric_names

        dataset_path.unlink()

    def test_save_and_load_report(self, session, repo):
        """Report can be saved to disk and loaded back."""
        service, chunk_ids = self._setup(session, repo)

        dataset = {
            "version": "save-test",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [
                {
                    "query": "simple query",
                    "relevant_chunk_ids": [chunk_ids[0]],
                    "tags": [],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dataset, f)
            dataset_path = Path(f.name)

        runner = EvaluationRunner(
            retrieval_service=service,
            dataset_path=dataset_path,
            top_k=5,
            k_values=[5],
        )

        with patch("retrieval.service.get_session", return_value=session):
            report = runner.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = EvaluationRunner.save_report(report, output_dir=Path(tmpdir))
            assert out_path.exists()
            assert out_path.suffix == ".json"

            # Verify it loads back
            loaded = EvaluationReport.model_validate_json(
                out_path.read_text(encoding="utf-8")
            )
            assert loaded.dataset_version == "save-test"

        dataset_path.unlink()

    def test_empty_dataset(self, session, repo):
        """Runner handles an empty dataset gracefully."""
        service, _ = self._setup(session, repo)

        dataset = {
            "version": "empty",
            "created_at": "2026-04-04T00:00:00Z",
            "annotations": [],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dataset, f)
            dataset_path = Path(f.name)

        runner = EvaluationRunner(
            retrieval_service=service,
            dataset_path=dataset_path,
            top_k=5,
        )

        with patch("retrieval.service.get_session", return_value=session):
            report = runner.run()

        assert len(report.per_query_results) == 0
        assert len(report.aggregate_metrics) == 0

        dataset_path.unlink()
```

##### Step 3 Verification Checklist
- [ ] No import errors: `python -c "from evaluation.retrieval_runner import EvaluationRunner; from evaluation.cli import main; print('OK')"`
- [ ] Module entry point works: `python -m evaluation --help`
- [ ] Runner tests pass: `python -m pytest test/test_evaluation_runner.py -v`
- [ ] All metric tests still pass: `python -m pytest test/test_retrieval_metrics.py -v`

#### Step 3 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.
```
git add evaluation/retrieval_runner.py evaluation/cli.py evaluation/__main__.py test/test_evaluation_runner.py && git commit -m "feat(evaluation): add evaluation runner and CLI"
```

---

### Step 4: Run baseline evaluation and update documentation

- [ ] Populate the ground-truth dataset with real chunk IDs.

Run this helper script to extract chunk IDs from the database for building ground-truth annotations:

```bash
python -c "
from database.connection import get_session
from database.models import ChunkModel, DocumentModel
from sqlalchemy import select

session = get_session()
docs = session.execute(select(DocumentModel)).scalars().all()
for doc in docs:
    print(f'\n=== {doc.title} ({doc.source_path}) ===')
    chunks = session.execute(
        select(ChunkModel)
        .where(ChunkModel.document_id == doc.id)
        .order_by(ChunkModel.position)
    ).scalars().all()
    for c in chunks:
        preview = c.text[:80].replace(chr(10), ' ')
        questions = c.hypothetical_questions[:2] if c.hypothetical_questions else []
        print(f'  ID: {c.id}')
        print(f'  Text: {preview}...')
        print(f'  Questions: {questions}')
        print()
session.close()
"
```

- [ ] Update `evaluation/datasets/retrieval_ground_truth.json` by replacing `REPLACE_WITH_REAL_CHUNK_ID_*` placeholders with actual chunk IDs from the output above. Use the `hypothetical_questions` as query inspiration.

- [ ] Run the baseline evaluation:

```bash
python -m evaluation --verbose
```

- [ ] Verify the report was saved to `evaluation/results/`.

- [ ] Update `README.md` — add an Evaluation section after the existing content:

Add this section to the end of `README.md`:

```markdown
## Evaluation

### Retrieval Evaluation

Measures retrieval quality using standard information retrieval metrics against a ground-truth dataset.

**Metrics:**
- **Precision@k** — Fraction of top-k results that are relevant
- **Recall@k** — Fraction of all relevant documents found in top-k
- **MRR** (Mean Reciprocal Rank) — Reciprocal of the rank of the first relevant result
- **NDCG@k** (Normalized Discounted Cumulative Gain) — Accounts for both relevance and ranking position

**Running:**
```bash
# Full evaluation with default settings
python -m evaluation

# With verbose per-query output
python -m evaluation --verbose

# Custom parameters
python -m evaluation --top-k 10 --k-values 1 3 5 10 --verbose

# Skip saving (dry run)
python -m evaluation --no-save --verbose
```

**Adding queries to the ground-truth dataset:**

Edit `evaluation/datasets/retrieval_ground_truth.json` and add new entries to the `annotations` array:
```json
{
  "query": "Your evaluation query here",
  "relevant_chunk_ids": ["chunk-uuid-1", "chunk-uuid-2"],
  "tags": ["factual"],
  "notes": "Why these chunks are relevant."
}
```

Use the chunk inspection script to find chunk IDs:
```bash
python -c "
from database.connection import get_session
from database.models import ChunkModel
from sqlalchemy import select
session = get_session()
for c in session.execute(select(ChunkModel).limit(20)).scalars():
    print(f'{c.id}: {c.text[:80]}...')
session.close()
"
```

**Reports** are saved to `evaluation/results/` as timestamped JSON files.
```

##### Step 4 Verification Checklist
- [ ] Ground-truth dataset has real chunk IDs (no `REPLACE_WITH_REAL_CHUNK_ID` placeholders remaining)
- [ ] Baseline evaluation runs without errors: `python -m evaluation --verbose`
- [ ] Report file exists in `evaluation/results/`
- [ ] Report contains all four metric types at all k values
- [ ] README instructions are accurate: `python -m evaluation --help` matches documented usage

#### Step 4 STOP & COMMIT
**STOP & COMMIT:** Agent must stop here and wait for the user to test, stage, and commit the change.
```
git add evaluation/datasets/ evaluation/results/ README.md && git commit -m "feat(evaluation): run baseline evaluation and add documentation"
```
