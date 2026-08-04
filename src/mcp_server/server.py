"""MCP server exposing HyridRAG's hybrid retrieval pipeline as a tool.

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

from src.ingestion import Embedder
from src.retrieval.dense import dense_search
from src.retrieval.fusion import Reranker, hybrid_retrieve
from src.retrieval.sparse import sparse_search
from src.config import settings

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


if __name__ == "__main__":
    mcp.run()
