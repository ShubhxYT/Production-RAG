"""Document upload endpoint with background ingestion pipeline."""

import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from pydantic import BaseModel

from database.connection import get_session
from database.repository import DocumentRepository
from embeddings.models import EmbeddingConfig
from embeddings.service import EmbeddingService
from ingestion.pipeline import IngestionPipeline
from observability.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# In-memory job store (thread-safe)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Accepted file extensions
ACCEPTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md"}
# 50 MB upload limit
MAX_FILE_SIZE = 50 * 1024 * 1024


class UploadJobResponse(BaseModel):
    job_id: str


class UploadStatusResponse(BaseModel):
    job_id: str
    stage: str
    progress: int
    error: str | None = None
    document_id: str | None = None


def _set_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _run_ingestion(job_id: str, tmp_path: str, original_name: str) -> None:
    """Full ingestion pipeline executed in a background thread."""

    def update(stage: str, progress: int) -> None:
        _set_job(job_id, stage=stage, progress=progress)
        logger.info("Upload job %s: %s (%d%%)", job_id, stage, progress)

    file_path = Path(tmp_path)

    try:
        # Stage 1 — Load, restructure, chunk
        update("loading", 20)
        pipeline = IngestionPipeline(output_dir=str(file_path.parent / "results"))
        doc = pipeline.ingest_file(file_path)
        if doc is None:
            raise ValueError(f"Unsupported or failed to process: {original_name}")

        # Allow original filename as title if not set by loader
        if doc.title is None:
            doc.title = Path(original_name).stem

        # Stage 2 — Save document & chunks to DB
        update("saving", 60)
        session = get_session()
        try:
            repo = DocumentRepository()
            repo.insert_document(session, doc)
            session.flush()

            # Stage 3 — Embed chunks
            update("embedding", 80)
            chunk_texts = [c.text for c in doc.chunks if c.text.strip()]
            chunk_ids = [c.id for c in doc.chunks if c.text.strip()]

            if chunk_texts:
                config = EmbeddingConfig()
                emb_service = EmbeddingService(config=config)
                result = emb_service.embed(chunk_texts)

                # Stage 4 — Store embeddings
                update("indexing", 95)
                embeddings = list(zip(chunk_ids, result.vectors))
                repo.insert_bulk_embeddings(session, embeddings, result.model)

            session.commit()
        finally:
            session.close()

        # Clean up temp file
        file_path.unlink(missing_ok=True)

        update("complete", 100)
        _set_job(job_id, document_id=doc.id)

    except Exception as exc:
        logger.exception("Upload job %s failed: %s", job_id, exc)
        _set_job(job_id, stage="error", progress=0, error=str(exc))
        file_path.unlink(missing_ok=True)


@router.post(
    "/upload",
    response_model=UploadJobResponse,
    status_code=202,
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> UploadJobResponse:
    """Accept a document file and start background ingestion.

    Returns a job_id that can be polled for progress.
    Accepted formats: PDF, DOCX, HTML, Markdown (max 50 MB).
    """
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS))}"
            ),
        )

    # Read and size-check
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 50 MB limit.",
        )

    # Write to a named temp file so the pipeline can read it by path
    suffix = ext
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "stage": "queued",
            "progress": 0,
            "error": None,
            "document_id": None,
        }

    logger.info(
        "Upload job %s queued: %s (%d bytes)",
        job_id,
        file.filename,
        len(content),
    )

    background_tasks.add_task(
        _run_ingestion, job_id, tmp_path, file.filename or "document"
    )
    return UploadJobResponse(job_id=job_id)


@router.get(
    "/upload/{job_id}/status",
    response_model=UploadStatusResponse,
)
def get_upload_status(job_id: str) -> UploadStatusResponse:
    """Poll the status of an upload/ingestion job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return UploadStatusResponse(
        job_id=job_id,
        stage=job["stage"],
        progress=job["progress"],
        error=job["error"],
        document_id=job["document_id"],
    )
