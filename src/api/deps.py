from fastapi import Request

from src.ingestion import Embedder
from src.retrieval.fusion import Reranker


# Thin, request-scoped accessors into app.state — kept as separate functions
# (instead of importing globals) so tests can override them with
# app.dependency_overrides[get_embedder] = lambda: FakeEmbedder().
def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_reranker(request: Request) -> Reranker:
    return request.app.state.reranker
