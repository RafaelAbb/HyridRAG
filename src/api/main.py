from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.ingestion import Embedder
from src.retrieval.fusion import Reranker


# Lifespan: build the expensive, stateful objects ONCE at startup, not per
# request. `ready` lets /health report 503 during the (short) startup window
# instead of routing traffic to a half-initialized app.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.embedder = Embedder()
    app.state.reranker = Reranker()
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
