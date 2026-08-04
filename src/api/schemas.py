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


class RetrieveRequest(BaseModel):
    # Raw-retrieval counterpart to AskRequest — stops before generation.
    # Same top_k bound as AskRequest: an unbounded k is a cost-control hole
    # the moment a caller (human or model, via the MCP tool) asks for too much.
    query: str = Field(min_length=1)
    top_k: int = Field(default=settings.reranker_top_k, ge=1, le=50)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID


class ChunkOut(BaseModel):
    # What's actually available at retrieval time — see dense.py/sparse.py,
    # metadata is flattened to source_name only (page/section/author are lost
    # upstream in embedder.py, a known gap, see future/README.md). Don't
    # promise fields here that the pipeline can't fill.
    doc_id: str
    text: str
    score: float
    source_name: str


class RetrieveResponse(BaseModel):
    chunks: list[ChunkOut]
    retrieval_mode: RetrievalMode


class AskResponse(BaseModel):
    answer: str
    has_answer: bool
    confidence: float
    sources: list[SourceOut]
    retrieval_mode: RetrievalMode


class IngestRequest(BaseModel):
    # Server-side path (file or directory) — secondary/power-user flow for
    # local testing (matches the CLI in main.py: point it at a path already
    # on the machine running the server, no upload round trip). The primary
    # UI-driven flow is POST /ingest/upload (see UploadIngestResponse below).
    path: str
    strategy: ChunkingStrategy = settings.default_chunk_strategy


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_created: int
    strategy: ChunkingStrategy


class SkippedFile(BaseModel):
    filename: str
    reason: str


class UploadIngestResponse(BaseModel):
    # Kept distinct from IngestResponse (not a shared/extended base) — a
    # partial-failure "skipped" list is meaningless for the path-based
    # /ingest, where a bad path is a hard 404/422, not a partial batch.
    documents_ingested: int
    chunks_created: int
    strategy: ChunkingStrategy
    skipped: list[SkippedFile] = []


class DocumentsResponse(BaseModel):
    sources: list[str]
    chunk_count: int
