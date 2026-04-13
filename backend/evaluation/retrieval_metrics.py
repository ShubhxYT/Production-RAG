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

    # IDCG: perfect ranking - all relevant items at the top
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
