# Hybrid RAG System — Backend

Production-style hybrid RAG (retrieval-augmented generation) pipeline: multi-format ingestion,
dense (ChromaDB) + sparse (BM25) retrieval fused with RRF and a cross-encoder reranker, grounded
generation with citation verification, and a FastAPI service in front of it. Built as a guided
learning project — see `CLAUDE.md` for the teaching context and lesson plan if you're picking up
where a prior session left off.

This file is written for whoever (human or AI agent) opens this repo next and needs to get
oriented fast.

## Architecture

```
Document (PDF / MD / HTML / TXT)
        │  src/ingestion/loader.py + loaders/*
        ▼
   RawDocument
        │  src/ingestion/chuncker.py + chunckers/*  (fixed | recursive | semantic)
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

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
uvicorn src.api.main:app --port 8000
```

**Do not use `--reload` on Windows in this environment** — it's been observed spawning the
reload worker under the global Python install instead of the project's venv, which appeared to
serve a stale copy of the app after edits (see `future/README.md`). Restart manually after
backend changes instead.

There's also a simple interactive CLI at `main.py` (insert documents, view the index, ask
questions) that talks to the same `Embedder`/retrieval/generation modules directly — it does not
go through the API, so it works even without the server running.

Interactive API docs: `http://localhost:8000/docs` once the server's up.

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness/readiness — 503 until startup (embedder + reranker) finishes |
| `/ask` | POST | `{question, top_k, retrieval_mode}` → grounded answer + citations + confidence. `retrieval_mode` is `hybrid` (default) \| `dense` \| `sparse` |
| `/ingest` | POST | `{path, strategy}` — ingest a file/folder already on the server's filesystem. Power-user/local-testing escape hatch (see `future/README.md` re: hardening if ever exposed beyond localhost) |
| `/ingest/upload` | POST | multipart: `files[]` + `relative_paths[]` + `strategy` — the primary UI-driven ingest flow, used by the dashboard's drag-and-drop |
| `/documents` | GET | List indexed source names + total chunk count |

Both ingest endpoints share the same underlying pipeline (`load_file`/`chunk_documents`/
`embedder.embed`) and the same idempotency guarantee: re-ingesting the same source (same path,
or same `relative_path` on upload) **upserts**, it doesn't duplicate — because
`Embedder.generate_id()` derives ChromaDB IDs from the file path. This is why `/ingest/upload`
saves files to a *stable* path under `data/uploads/<relative_path>` rather than a random temp
dir — see the comment at the save step in `src/api/routes.py`.

## Config

All settings live in `src/config.py` (`pydantic-settings`, loaded from `.env` — see
`.env.example` for the full list with descriptions). Config validates at import time: a missing
`OPENAI_API_KEY` crashes loudly at startup, not silently mid-run.

Notable ones if you're new to this codebase:
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
│   ├── chuncker.py         ← chunk_document()/chunk_documents() (factory dispatch)
│   ├── chunckers/          ← fixed / recursive / semantic chunking strategies
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
future/                      ← backlog of deferred ideas/hardening/bugs — check before starting new work
data/                        ← gitignored: data/chroma/ (vector index), data/uploads/ (ingested files)
tests/                       ← pytest unit tests; tests/llm/ + tests/test_deepeval_eval.py make real LLM calls (marker: llm_eval, excluded by default)
```

## Key design decisions worth knowing before you change things

- **Idempotency via path-derived IDs.** `Embedder.generate_id()` builds ChromaDB upsert IDs from
  the full file path a loader was given. Anything that changes how files land on disk (new
  ingest path, new upload flow, etc.) must preserve stable, repeatable paths or re-ingestion will
  silently duplicate instead of overwrite.
- **Graceful degradation in retrieval.** `hybrid_retrieve()` falls back to plain RRF order if the
  cross-encoder reranker throws — don't let a reranker failure take down `/ask` entirely.
- **`/ask`'s citation contract.** `generate_answer()`'s prompt numbers retrieved chunks
  `[1]`, `[2]`, ... (1-based, by position — not by `doc_id`). The frontend's citation-chip
  rendering depends on this exact convention; see the comment in `frontend/src/api/client.js`.
- **No authentication anywhere.** Deliberate for a single-user local/portfolio tool. Don't expose
  this beyond localhost without addressing `future/README.md`'s backend security items first.
