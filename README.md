# Hybrid RAG System

Hybrid RAG (retrieval-augmented generation) service over internal documents: multi-format
ingestion, dense (ChromaDB) + sparse (BM25) retrieval fused with RRF and a cross-encoder
reranker, grounded generation with inline citations and LLM-as-judge verification, a FastAPI
backend, a React dashboard, and a RAGAS evaluation harness — all Dockerized.

This file is written for whoever (human or AI agent) opens this repo next and needs to get
oriented fast. See `CLAUDE.md` for engineering context, conventions, and current status.

## Eval results

RAGAS scores on a 30-question hand-written golden dataset (never LLM-generated —
see `evals/ragas/golden_dataset.json`), latest run `evals/results/20260728T195847Z.json`:

| Metric | Score | |
|---|---|---|
| Faithfulness | **0.87** | generated claims are grounded in retrieved context |
| Answer relevancy | **0.85** | answers actually address the question asked |
| Context precision | **0.23** | ⚠️ retrieved context is often not the most relevant available |
| Context recall | **0.42** | ⚠️ retrieval frequently misses relevant golden context entirely |

Generation is solid; retrieval precision/recall is the known weak point and the active work
item. `evals/ragas/diagnose_precision.py` does post-hoc failure analysis — for each flagged
question it prints golden vs. actually-retrieved contexts side by side and resolves each
snippet back to its source file, so "wrong file entirely" is distinguishable from "right file,
wrong chunk" at a glance. See `future/README.md` before starting new retrieval work.

Reproduce: `python -m evals.ragas.run_eval` (writes a new timestamped result to `evals/results/`).

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

**Do not use `--reload` on Windows in this environment** — it's been observed spawning the
reload worker under the global Python install instead of the project's venv, which appeared to
serve a stale copy of the app after edits (see `future/README.md`). Restart manually after
backend changes instead.

There's also a simple interactive CLI at `main.py` (insert documents, view the index, ask
questions) that talks to the same `Embedder`/retrieval/generation modules directly — it does not
go through the API, so it works even without the server running.

Interactive API docs: `http://localhost:8000/docs` once the server's up.

## MCP server

`src/mcp_server/server.py` exposes the retrieval pipeline as an [MCP](https://modelcontextprotocol.io)
tool — `retrieve(query, k, mode)` — for any MCP-aware host (Claude Desktop, Claude Code, a future
agent project) to call directly, without going through `/ask`'s full answer-generation flow. It
returns raw ranked chunks (`doc_id`, `text`, `score`, `source_name`), not a generated answer —
built for a consumer that wants to reason over the material itself.

It runs as its own stdio subprocess, independent of the FastAPI app — it builds its own
`Embedder`/`Reranker` in-process at startup (same pattern as `main.py`'s CLI), so it works
whether or not `uvicorn` is running.

```bash
# Local dev / inspector
mcp dev src/mcp_server/server.py

# Connect it to Claude Code (this repo, from the repo root)
claude mcp add hyridrag-retrieval -- "<path-to-venv-python>" -m src.mcp_server.server

# Or Claude Desktop — add to claude_desktop_config.json:
#   "hyridrag-retrieval": {
#     "command": "<path-to-venv-python>",
#     "args": ["-m", "src.mcp_server.server"]
#   }
```

The tool's docstring is the *only* documentation the calling model ever sees — it's written to
be precise about what's returned (and what isn't: no page/section metadata, see
`future/README.md`) rather than vague, since that's what determines whether the model calls it
well.

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness/readiness — 503 until startup (embedder + reranker) finishes |
| `/ask` | POST | `{question, top_k, retrieval_mode}` → grounded answer + citations + confidence. `retrieval_mode` is `hybrid` (default) \| `dense` \| `sparse` |
| `/retrieve` | POST | `{query, top_k, retrieval_mode}` → raw ranked chunks, no generation. Same pipeline as `/ask` up to the point it would hand off to the generator — the HTTP counterpart to the MCP `retrieve` tool below |
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
├── api/
│   ├── main.py              ← FastAPI() + lifespan (builds Embedder/Reranker once) + CORS + router include
│   ├── deps.py               ← get_embedder()/get_reranker() — thin app.state accessors for Depends()
│   ├── schemas.py             ← every request/response Pydantic model
│   └── routes.py               ← all endpoint handlers
└── mcp_server/
    └── server.py            ← FastMCP server, `retrieve` tool, own Embedder/Reranker, stdio transport

frontend/                    ← React dashboard, see frontend/README.md
future/                      ← backlog of deferred ideas/hardening/bugs — check before starting new work
data/                        ← gitignored: data/chroma/ (vector index), data/uploads/ (ingested files)
evals/                       ← RAGAS harness, golden dataset, diagnostic tools, timestamped results
tests/                       ← pytest unit tests; llm_eval-marked tests make real LLM calls, excluded by default
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
