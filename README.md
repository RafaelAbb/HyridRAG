# Hybrid RAG System

Hybrid RAG (retrieval-augmented generation) service: multi-format ingestion, dense
(ChromaDB) + sparse (BM25) retrieval fused with RRF and a cross-encoder reranker,
grounded generation with inline citations and LLM-as-judge verification, a FastAPI
backend, a React dashboard, and a RAGAS evaluation harness — all Dockerized.

## Eval results

RAGAS scores on a 30-question hand-written golden dataset (never LLM-generated —
see `evals/ragas/golden_dataset.json`), latest run `evals/results/20260728T195847Z.json`,
against a public technical-docs corpus (the FastAPI documentation site) chosen so the
results are reproducible without proprietary data:

| Metric | Score | |
|---|---|---|
| Faithfulness | **0.87** | generated claims are grounded in retrieved context |
| Answer relevancy | **0.85** | answers actually address the question asked |
| Context precision | **0.23** | retrieved context is often not the most relevant available |
| Context recall | **0.42** | retrieval frequently misses relevant golden context entirely |

Generation is solid; retrieval precision/recall is the known weak point, probably due to a
weak eval dataset.

Reproduce: `python -m evals.ragas.run_eval` (writes a new timestamped result to `evals/results/`).

## Architecture

```
Document (PDF / MD / HTML / TXT)
        │  src/ingestion/loader.py + loaders/*
        ▼
   RawDocument
        │  src/ingestion/chunker.py + chunkers/*  (fixed | recursive | semantic)
        ▼
    Chunk[]
        │  src/ingestion/embedder.py
   ┌────┴─────────────────────┐
   │ ChromaDB (dense vectors) │  +  BM25 index (sparse, built from Chroma's own store)
   └────┬─────────────────────┘
        │  query time
   ┌────┴────┐   ┌──────────────┐
   │dense.py │   │  sparse.py   │
   └────┬────┘   └──────┬───────┘
        └──────┬─────────┘
               ▼  fusion.py  (RRF → cross-encoder rerank, graceful degradation on rerank failure)
          top-k chunks
               │
               ▼  src/generation/generator.py
     answer + citations + LLM-as-judge verification + confidence
               │
               ▼  src/api/  (FastAPI)
     GET /health   POST /ask   POST /ingest   POST /ingest/upload   GET /documents
```

## Running it

**Docker (both services, one command):**

```bash
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
docker compose up --build
```

Backend on `:8000`, frontend on `:5173`.

**Locally:**

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
uvicorn src.api.main:app --port 8000
```

Interactive API docs: `http://localhost:8000/docs` once the server's up.

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness/readiness — 503 until startup (embedder + reranker) finishes |
| `/ask` | POST | `{question, top_k, retrieval_mode}` → grounded answer + citations + confidence. `retrieval_mode` is `hybrid` (default) \| `dense` \| `sparse` |
| `/retrieve` | POST | `{query, top_k, retrieval_mode}` → raw ranked chunks, no generation |
| `/ingest` | POST | `{path, strategy}` — ingest a file/folder already on the server's filesystem |
| `/ingest/upload` | POST | multipart: `files[]` + `relative_paths[]` + `strategy` — the primary UI-driven ingest flow, used by the dashboard's drag-and-drop |
| `/documents` | GET | List indexed source names + total chunk count |

Both ingest endpoints share the same underlying pipeline and the same idempotency guarantee:
re-ingesting the same source **upserts**, it doesn't duplicate.

## Config

All settings live in `src/config.py` (`pydantic-settings`, loaded from `.env` — see
`.env.example` for the full list). Config validates at import time: a missing
`OPENAI_API_KEY` crashes loudly at startup, not silently mid-run.

Notable settings:
- `default_chunk_strategy` — `fixed` \| `recursive` \| `semantic`
- `cors_origins` — comma-separated, defaults to the Vite dev server's `localhost:5173`
- `upload_dir` / `max_upload_size_mb` — for `/ingest/upload`

## Folder map

```
src/
├── config.py               ← Settings (pydantic-settings), ChunkingStrategy enum
├── ingestion/
│   ├── base.py             ← RawDocument, Chunk, DocumentMetadata, Source dataclasses; interfaces
│   ├── loader.py           ← load_file()/load_directory(), FileExtension enum (factory dispatch)
│   ├── loaders/            ← PDFLoader, MarkdownLoader, HTMLLoader, TextLoader (one per format)
│   ├── chunker.py          ← chunk_document()/chunk_documents() (factory dispatch)
│   ├── chunkers/           ← fixed / recursive / semantic chunking strategies
│   └── embedder.py         ← Embedder: OpenAI embeddings → ChromaDB upsert, batched, idempotent
├── retrieval/
│   ├── base.py             ← RetrievalResult dataclass
│   ├── dense.py            ← dense_search() — ChromaDB cosine NN
│   ├── sparse.py           ← sparse_search() — BM25 over Chroma's own stored documents
│   └── fusion.py           ← rrf_merge(), Reranker class, hybrid_retrieve() entry point
├── generation/
│   ├── base.py             ← GenerationResult, CitationVerification, JudgeEnum
│   └── generator.py        ← generate_answer(), judge_citations() (LLM-as-judge)
└── api/
    ├── main.py              ← FastAPI() + lifespan (builds Embedder/Reranker once) + CORS + router include
    ├── deps.py               ← get_embedder()/get_reranker() — thin app.state accessors for Depends()
    ├── schemas.py             ← every request/response Pydantic model
    └── routes.py               ← all endpoint handlers

frontend/                    ← React dashboard, see frontend/README.md
future/                      ← backlog of deferred ideas/hardening/bugs
data/                        ← gitignored: data/chroma/ (vector index), data/uploads/ (ingested files)
evals/                       ← RAGAS harness, golden dataset, diagnostic tools, timestamped results
tests/                       ← pytest unit tests; llm_eval-marked tests make real LLM calls, excluded by default
```

## License

MIT — see [LICENSE](LICENSE).
