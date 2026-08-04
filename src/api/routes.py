import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from src.api.deps import get_embedder, get_reranker
from src.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    DocumentsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrievalMode,
    RetrieveRequest,
    RetrieveResponse,
    SkippedFile,
    SourceOut,
    UploadIngestResponse,
)
from src.config import ChunkingStrategy, settings
from src.ingestion import Embedder, chunk_documents, load_directory, load_file
from src.ingestion.loader import FileExtension
from src.retrieval.dense import dense_search
from src.retrieval.sparse import sparse_search
from src.retrieval.fusion import Reranker, hybrid_retrieve
from src.generation.generator import generate_answer

router = APIRouter()


def _retrieve_by_mode(
    query: str,
    mode: RetrievalMode,
    embedder: Embedder,
    reranker: Reranker,
    k: int,
):
    """Shared dense/sparse/hybrid dispatch — used by both /ask (which goes on
    to generate a prose answer) and /retrieve (which stops here and hands
    the raw chunks back, e.g. to the MCP tool)."""
    match mode:
        case RetrievalMode.SPARSE:
            return sparse_search(query, embedder.collection, k=k)
        case RetrievalMode.DENSE:
            return dense_search(query, embedder.collection, k=k)
        case _:
            return hybrid_retrieve(query, embedder.collection, reranker, k=k)


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request):
    if not request.app.state.ready:
        # 503, not 200+false: health checks (Docker/k8s/load balancers) key
        # off the status code, not the body.
        raise HTTPException(status_code=503, detail="Service not ready")
    return HealthResponse(status="ok")


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    embedder: Embedder = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker),
):
    retrieved = _retrieve_by_mode(
        payload.question, payload.retrieval_mode, embedder, reranker, payload.top_k
    )

    if not retrieved:
        # Nothing indexed yet is a valid state, not a server error — 200
        # with has_answer=False, same contract as an unanswerable question.
        return AskResponse(
            answer="", has_answer=False, confidence=0.0, sources=[],
            retrieval_mode=payload.retrieval_mode,
        )

    result = generate_answer(payload.question, retrieved)

    sources = [
        SourceOut(
            doc_id=r.doc_id,
            source_name=r.metadata.source.source_name if r.metadata and r.metadata.source else "unknown",
            score=r.score,
        )
        for r in retrieved
    ]

    return AskResponse(
        answer=result.answer,
        has_answer=result.has_answer,
        confidence=result.confidence,
        sources=sources,
        retrieval_mode=payload.retrieval_mode,
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    payload: RetrieveRequest,
    embedder: Embedder = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker),
):
    """Raw retrieval, no generation — the material, not an opinion about it.
    Same pipeline /ask uses up to the point it hands off to generate_answer();
    this stops there. Exists for callers (the MCP tool, a future agent) that
    want to do their own reasoning over the chunks instead of relaying a
    pre-written answer."""
    retrieved = _retrieve_by_mode(
        payload.query, payload.retrieval_mode, embedder, reranker, payload.top_k
    )

    chunks = [
        ChunkOut(
            doc_id=r.doc_id,
            text=r.text,
            score=r.score,
            source_name=r.metadata.source.source_name if r.metadata and r.metadata.source else "unknown",
        )
        for r in retrieved
    ]

    return RetrieveResponse(chunks=chunks, retrieval_mode=payload.retrieval_mode)


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    payload: IngestRequest,
    embedder: Embedder = Depends(get_embedder),
):
    if not os.path.exists(payload.path):
        raise HTTPException(status_code=404, detail=f"Path not found: {payload.path}")

    raw_documents = (
        load_directory(payload.path) if os.path.isdir(payload.path) else load_file(payload.path)
    )
    if not raw_documents:
        raise HTTPException(status_code=422, detail="No loadable documents found at path")

    chunks = chunk_documents(raw_documents, strategy=payload.strategy)
    embedder.embed(chunks)

    return IngestResponse(
        documents_ingested=len(raw_documents),
        chunks_created=len(chunks),
        strategy=payload.strategy,
    )


def _sanitize_relative_path(relative_path: str) -> str | None:
    """Normalize an upload's relative path and guard against traversal
    outside upload_dir. Returns None if the path is unsafe."""
    normalized = os.path.normpath(relative_path.lstrip("/\\"))
    if normalized.startswith("..") or os.path.isabs(normalized):
        return None
    return normalized


@router.post("/ingest/upload", response_model=UploadIngestResponse)
def ingest_upload(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(...),
    strategy: ChunkingStrategy = Form(default=settings.default_chunk_strategy),
    embedder: Embedder = Depends(get_embedder),
):
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    supported_extensions = {e.name for e in FileExtension}

    saved_paths: list[str] = []
    skipped: list[SkippedFile] = []
    total_bytes = 0

    for file, relative_path in zip(files, relative_paths):
        # Stable path (not tempfile.mkdtemp()) is required for upsert
        # idempotency — Embedder.generate_id() derives chunk IDs from the
        # full path a loader was given, so the same relative_path must
        # always land at the same saved_path on re-upload.
        sanitized = _sanitize_relative_path(relative_path)
        if sanitized is None:
            skipped.append(SkippedFile(filename=relative_path, reason="Invalid or unsafe path"))
            continue

        extension = os.path.splitext(sanitized)[1].lstrip(".").upper()
        if extension not in supported_extensions:
            skipped.append(SkippedFile(filename=relative_path, reason=f"Unsupported extension: .{extension.lower()}"))
            continue

        # Size check before writing bytes — read the SpooledTemporaryFile
        # directly (sync def handler: UploadFile's async .read() can't be
        # awaited here, and calling it unawaited would silently no-op).
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
        total_bytes += size
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload batch exceeds the {settings.max_upload_size_mb}MB limit",
            )

        saved_path = os.path.join(settings.upload_dir, sanitized)
        os.makedirs(os.path.dirname(saved_path) or settings.upload_dir, exist_ok=True)
        with open(saved_path, "wb") as f:
            f.write(file.file.read())
        saved_paths.append(saved_path)

    raw_documents = []
    for path in saved_paths:
        try:
            raw_documents.extend(load_file(path))
        except ValueError as e:
            skipped.append(SkippedFile(filename=path, reason=str(e)))

    if not raw_documents:
        raise HTTPException(status_code=422, detail="No loadable documents found in this upload")

    chunks = chunk_documents(raw_documents, strategy=strategy)
    embedder.embed(chunks)

    return UploadIngestResponse(
        documents_ingested=len(raw_documents),
        chunks_created=len(chunks),
        strategy=strategy,
        skipped=skipped,
    )


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(
    embedder: Embedder = Depends(get_embedder),
):
    results = embedder.collection.get(include=["metadatas"])
    sources = sorted({m.get("source", "unknown") for m in results["metadatas"]})
    return DocumentsResponse(sources=sources, chunk_count=len(results["ids"]))
