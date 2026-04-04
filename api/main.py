"""FastAPI application for the FullRag RAG system."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import documents, health, query
from config.settings import get_log_file, get_log_format, get_log_level
from observability.logging import configure_logging, get_logger

configure_logging(
    level=get_log_level(),
    fmt=get_log_format(),
    log_file=get_log_file(),
)
logger = get_logger(__name__)

app = FastAPI(
    title="FullRag API",
    description="RAG pipeline REST API - retrieve and generate grounded answers.",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["health"])
app.include_router(documents.router, tags=["documents"])
app.include_router(query.router, tags=["query"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return structured JSON errors."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred."},
    )
