"""Query endpoint - main RAG pipeline entry point."""

import logging

from fastapi import APIRouter, HTTPException

from api.models import ErrorResponse, QueryRequest, QueryResponse
from pipeline.rag import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-initialized pipeline singleton
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from config.settings import get_generation_provider as get_provider_name

        _pipeline = RAGPipeline(provider_name=get_provider_name())
    return _pipeline


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def query_rag(request: QueryRequest) -> QueryResponse:
    """Execute the RAG pipeline and return a grounded answer.

    Args:
        request: Query request with question and optional parameters.

    Returns:
        QueryResponse with answer, sources, latency, and token usage.
    """
    logger.info(
        "Query received: question='%s' top_k=%d variant=%s",
        request.question[:80],
        request.top_k,
        request.prompt_variant,
    )

    try:
        pipeline = _get_pipeline()
        rag_response = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            prompt_variant=request.prompt_variant,
        )
    except Exception as e:
        logger.error("RAG pipeline error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    logger.info(
        "Query complete: answer_len=%d sources=%d total_ms=%.1f",
        len(rag_response.answer),
        len(rag_response.sources),
        rag_response.latency.total_ms,
    )

    return QueryResponse(
        answer=rag_response.answer,
        sources=rag_response.sources,
        latency=rag_response.latency,
        token_usage=rag_response.token_usage,
        prompt_version=rag_response.prompt_version,
    )
