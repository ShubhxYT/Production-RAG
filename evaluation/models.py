"""Pydantic models for evaluation datasets, metrics, and reports."""

from enum import Enum
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


# ---------------------------------------------------------------------------
# Generation evaluation models
# ---------------------------------------------------------------------------


class JudgeDimension(str, Enum):
    """Dimensions for LLM-based generation evaluation."""

    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"


class JudgeScore(BaseModel):
    """Score from a single judge dimension."""

    dimension: JudgeDimension = Field(description="Which quality dimension was evaluated.")
    score: int = Field(ge=1, le=5, description="Score from 1 (worst) to 5 (best).")
    reasoning: str = Field(description="Explanation for the score.")
    passed: bool = Field(description="Whether the score meets the passing threshold (>= 3).")


class GenerationEvalResult(BaseModel):
    """Result of evaluating a single generation with all judges."""

    query: str = Field(description="The query that was evaluated.")
    answer: str = Field(description="The generated answer.")
    context_chunks: list[str] = Field(
        default_factory=list, description="Context chunk texts used for generation."
    )
    scores: list[JudgeScore] = Field(
        default_factory=list, description="Scores from each judge dimension."
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata.")


class GenerationAnnotation(BaseModel):
    """A single query for generation evaluation."""

    query: str = Field(description="The evaluation query.")
    expected_answer: str = Field(
        default="",
        description="Optional expected/reference answer.",
    )
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="UUIDs of chunks relevant to this query.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Query category tags.",
    )


class GenerationGroundTruth(BaseModel):
    """A versioned dataset of generation evaluation queries."""

    version: str = Field(description="Semantic version of this dataset.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this dataset was created.",
    )
    annotations: list[GenerationAnnotation] = Field(
        default_factory=list,
        description="List of generation evaluation queries.",
    )


class GenerationQueryResult(BaseModel):
    """Per-query generation evaluation result."""

    query: str = Field(description="The query text.")
    generated_answer: str = Field(description="The generated answer.")
    sources: list[str] = Field(
        default_factory=list, description="Source document paths."
    )
    judge_scores: list[JudgeScore] = Field(
        default_factory=list, description="Scores from each judge."
    )
    latency_ms: float = Field(default=0.0, description="Total pipeline latency.")
    token_usage: dict = Field(
        default_factory=dict, description="Token usage from generation."
    )


class GenerationEvaluationReport(BaseModel):
    """Full generation evaluation report."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evaluation was run.",
    )
    dataset_version: str = Field(description="Version of the dataset used.")
    pipeline_config: dict = Field(
        default_factory=dict, description="Pipeline configuration snapshot."
    )
    aggregate_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Average score per dimension.",
    )
    per_query_results: list[GenerationQueryResult] = Field(
        default_factory=list,
        description="Per-query results with judge scores.",
    )
    pass_rate: dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of queries passing per dimension.",
    )
