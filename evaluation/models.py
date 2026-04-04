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
