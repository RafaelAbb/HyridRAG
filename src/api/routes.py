import os

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.deps import get_embedder, get_reranker
from src.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrievalMode,
    SourceOut,
)
from src.ingestion import Embedder, chunk_documents, load_directory, load_file
from src.retrieval.dense import dense_search
from src.retrieval.sparse import sparse_search
from src.retrieval.fusion import Reranker, hybrid_retrieve
from src.generation.generator import generate_answer

router = APIRouter()


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
    
    match payload.retrieval_mode:
        case RetrievalMode.SPARSE:
            retrieved = sparse_search(payload.question, embedder.collection, k=payload.top_k)
        
        case RetrievalMode.DENSE:
            retrieved = dense_search(payload.question, embedder.collection, k=payload.top_k)
            
        case _:
            retrieved = hybrid_retrieve(payload.question, embedder.collection, reranker, k=payload.top_k)
    
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


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(
    embedder: Embedder = Depends(get_embedder),
):
    results = embedder.collection.get(include=["metadatas"])
    sources = sorted({m.get("source", "unknown") for m in results["metadatas"]})
    return DocumentsResponse(sources=sources, chunk_count=len(results["ids"]))
