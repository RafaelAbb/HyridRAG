from enum import Enum

from pydantic import BaseModel, Field

from src.config import ChunkingStrategy, settings


class HealthResponse(BaseModel):
    status: str


class RetrievalMode(str, Enum):
    # API-layer concept, not a pipeline setting — dense.py/fusion.py don't
    # need to know "modes" exist, only the /ask route branches on this.
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=settings.reranker_top_k, ge=1, le=50)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID


class SourceOut(BaseModel):
    doc_id: str
    source_name: str
    score: float


class AskResponse(BaseModel):
    answer: str
    has_answer: bool
    confidence: float
    sources: list[SourceOut]
    retrieval_mode: RetrievalMode


class IngestRequest(BaseModel):
    # Server-side path (file or directory) — this is a local/portfolio tool,
    # not a multi-tenant SaaS, so "point it at a path on disk" matches the
    # CLI in main.py and avoids building multipart upload handling that
    # nothing in the spec actually calls for.
    path: str
    strategy: ChunkingStrategy = settings.default_chunk_strategy


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    strategy: ChunkingStrategy


class DocumentsResponse(BaseModel):
    sources: list[str]
    chunk_count: int
