"""Query endpoint - main RAG pipeline entry point."""

from fastapi import APIRouter, HTTPException
from database.connection import get_session
from database.repository import DocumentRepository
from observability.logging import get_logger
from observability.logging import get_request_id

from api.models import ErrorResponse, QueryRequest, QueryResponse
from pipeline.rag import RAGPipeline

logger = get_logger(__name__)
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
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Execute the RAG pipeline and return a grounded answer.

    Args:
        request: Query request with question and optional parameters.

    Returns:
        QueryResponse with answer, sources, latency, and token usage.
    """
    logger.info(
        "Query received",
        extra={
            "question": request.question[:80],
            "top_k": request.top_k,
            "variant": request.prompt_variant,
        },
    )

    try:
        pipeline = _get_pipeline()
        rag_response = await pipeline.query(
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
        "Query complete",
        extra={
            "answer_len": len(rag_response.answer),
            "sources": len(rag_response.sources),
            "total_ms": rag_response.latency.total_ms,
        },
    )

    # Audit log (fire-and-forget, don't block response)
    try:
        session = get_session()
        try:
            repo = DocumentRepository()
            repo.insert_query_log(session, {
                "request_id": get_request_id(),
                "query": request.question,
                "answer": rag_response.answer,
                "sources": [s.model_dump() for s in rag_response.sources],
                "prompt_variant": request.prompt_variant,
                "prompt_version": rag_response.prompt_version,
                "retrieval_top_k": request.top_k,
                "retrieval_result_count": len(rag_response.sources),
                "latency_ms": rag_response.latency.total_ms,
                "retrieval_ms": rag_response.latency.retrieval_ms,
                "generation_ms": rag_response.latency.generation_ms,
                "prompt_tokens": rag_response.token_usage.prompt_tokens,
                "completion_tokens": rag_response.token_usage.completion_tokens,
                "model": rag_response.prompt_version,
            })
            session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.warning("Failed to insert query log: %s", e)

    return QueryResponse(
        answer=rag_response.answer,
        sources=rag_response.sources,
        latency=rag_response.latency,
        token_usage=rag_response.token_usage,
        prompt_version=rag_response.prompt_version,
    )
