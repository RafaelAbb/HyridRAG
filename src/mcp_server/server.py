"""MCP server exposing HyridRAG's hybrid retrieval pipeline as MCP tools.

Runs over stdio — the transport Claude Desktop/Code use for local servers,
and the simplest one to get right first (see CLAUDE.md / this file's design
notes). It does NOT go through the FastAPI app or HTTP at all: an MCP host
launches this as its own subprocess with no guarantee the API server is
running, so it builds its own Embedder/Reranker in-process at startup — the
same pattern `main.py`'s CLI already uses, for the same reason.

(Package is `src.mcp_server`, not `src.mcp` — deliberately, so nothing here
ever shadows the third-party `mcp` SDK this file imports from.)

Local dev / inspector:
    mcp dev src/mcp_server/server.py

Point a real host at it (stdio):
    python -m src.mcp_server.server
"""

import os
import sys
from pathlib import Path

# Anchor cwd to the repo root regardless of what directory the MCP host
# launches this subprocess from. settings.chroma_persist_dir etc. are
# relative paths (./data/chroma) — launched from the wrong cwd, this
# silently opens a different (or empty) Chroma store instead of failing
# loudly, which is exactly the kind of bug that's hard to notice from a
# host's tool-call UI.
REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from src.config import ChunkingStrategy, settings
from src.ingestion import Embedder, chunk_documents, load_directory, load_file
from src.retrieval.dense import dense_search
from src.retrieval.fusion import Reranker, hybrid_retrieve
from src.retrieval.sparse import sparse_search

mcp = FastMCP("hyridrag-retrieval")

# Built once at process startup, not per call — constructing a fresh
# Reranker() per call would reload the cross-encoder model on every tool
# call (the exact mistake the API's lifespan hook avoids; see
# src/api/main.py).
_embedder = Embedder()
_reranker = Reranker()

# Tighter than the /retrieve HTTP route's bound (top_k, le=50): a model
# deciding k for itself is a sharper cost-control risk than a human typing
# a number into a form, so the ceiling here is stricter.
MAX_K = 20


@mcp.tool()
def retrieve(query: str, k: int = settings.reranker_top_k, mode: str = "hybrid") -> list[dict]:
    """Search the user's indexed documents for passages relevant to a query.

    Returns up to `k` ranked excerpts — source name, a relevance score, and
    the raw chunk text. Does NOT generate an answer, verify citations, or
    interpret the material in any way. Call this when you need the
    underlying source material to reason over yourself, not a finished
    response to relay.

    Args:
        query: The question or topic to search for.
        k: Max number of chunks to return. Bounded to 20 regardless of what
            is requested; omit to use the server's configured default.
        mode: "hybrid" (dense + sparse fused and reranked — default, best
            general-purpose quality), "dense" (semantic/embedding search
            only), or "sparse" (BM25 keyword search only — better than
            dense for exact strings, codes, or acronyms).

    Returns:
        A list of chunks, each shaped:
        {"doc_id": str, "text": str, "score": float, "source_name": str}.
        Note: only the source document's name is available, not page number
        or section — that metadata isn't currently tracked past ingestion
        (see future/README.md).
    """
    k = max(1, min(k, MAX_K))

    match mode:
        case "dense":
            results = dense_search(query, _embedder.collection, k=k)
        case "sparse":
            results = sparse_search(query, _embedder.collection, k=k)
        case _:
            results = hybrid_retrieve(query, _embedder.collection, _reranker, k=k)

    return [
        {
            "doc_id": r.doc_id,
            "text": r.text,
            "score": r.score,
            "source_name": (
                r.metadata.source.source_name if r.metadata and r.metadata.source else "unknown"
            ),
        }
        for r in results
    ]


@mcp.tool()
def ingest(path: str, strategy: str = settings.default_chunk_strategy.value) -> dict:
    """Ingest a file or folder into the index so `retrieve` can find it.

    `path` is resolved on the machine THIS SERVER runs on, not the caller's —
    same as the API's POST /ingest (see src/api/routes.py), and the same
    deliberate tradeoff: no path restriction, no auth, because this is a
    single-user local tool. Don't wire this tool up to a host that isn't
    trusted with arbitrary filesystem reads on this machine (see
    future/README.md before ever exposing either beyond localhost).

    Re-ingesting the same path is safe to call again — it upserts rather
    than duplicating, because chunk IDs are derived from the file path
    (see Embedder.generate_id() in src/ingestion/embedder.py).

    Args:
        path: Absolute path to a file or directory, already on this machine.
        strategy: Chunking strategy — "fixed", "recursive", or "semantic".
            Defaults to the server's configured default.

    Returns:
        On success: {"documents_ingested": int, "chunks_created": int,
        "strategy": str}. On failure (bad path, unsupported/empty
        directory, unknown strategy): {"error": str} — never raises, so a
        bad call is something the calling model can see and react to.
    """
    if not os.path.exists(path):
        return {"error": f"Path not found: {path}"}

    try:
        chunk_strategy = ChunkingStrategy(strategy)
    except ValueError:
        valid = ", ".join(s.value for s in ChunkingStrategy)
        return {"error": f"Unknown strategy '{strategy}', expected one of: {valid}"}

    raw_documents = load_directory(path) if os.path.isdir(path) else load_file(path)
    if not raw_documents:
        return {"error": "No loadable documents found at path"}

    chunks = chunk_documents(raw_documents, strategy=chunk_strategy)
    _embedder.embed(chunks)

    return {
        "documents_ingested": len(raw_documents),
        "chunks_created": len(chunks),
        "strategy": chunk_strategy.value,
    }


if __name__ == "__main__":
    mcp.run()
