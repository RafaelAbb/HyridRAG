# CLAUDE.md — Hybrid RAG System

> Read by Claude at the start of every session. This is an engineering context file
> for a production-style codebase — not a lesson plan. There is no teaching contract
> here; treat requests the way you would on any other production repo (plan before
> large changes, ask before destructive ones, don't touch what wasn't asked for).

---

## What this project is

A hybrid retrieval-augmented generation service over internal documents. Portfolio
project (Project 1 of 3 in Rafael's AI Engineer portfolio — see profile below) but
built and operated as production code: real eval numbers, Docker deploy, no
shortcuts taken because "it's just an exercise."

- Multi-format document ingestion (PDF, Markdown, HTML, TXT)
- Three chunking strategies: fixed-size, recursive, semantic
- Dual indexing: ChromaDB (dense / semantic) + BM25 (sparse / keyword)
- Reciprocal Rank Fusion (RRF) + cross-encoder reranker, with graceful degradation
- Grounded generation with inline citations and LLM-as-judge verification
- Confidence scoring
- RAGAS evaluation harness against a 30-question hand-written golden dataset
- FastAPI backend + React dashboard, both Dockerized

**Tech stack:** Python 3.12, OpenAI API, ChromaDB, rank-bm25, langchain-text-splitters,
FastAPI, uvicorn, pydantic-settings, PyMuPDF, sentence-transformers, RAGAS, pytest,
React 19 + Vite (frontend), Docker/docker-compose, MCP Python SDK (`mcp[cli]`, pinned to
1.29.0 — 2.0.0 restructured `FastMCP`'s API).

**Repo:** https://github.com/RafaelAbb/HyridRAG — branch `main`.

---

## Who Rafael is

- Software Engineer with ~3 years production experience at Rafael Advanced Defense
  Systems (.NET Framework 4.7.2, C#, async/event-driven architecture, hardware
  integration)
- B.Sc. Software Engineering, Ort Braude College (2025)
- ML background: capstone used YOLOv8 + Attention U-Net in PyTorch, published to
  IFIP AIAI 2026 (mAP 0.79, Dice 0.97)
- Known stack: C#, Python, FastAPI, MongoDB, React, Git, Azure DevOps
- Pivoting into AI Engineering (Path B, BASWE AI Pivot Field Guide)
- Often uses voice-to-text — messages may have typos or garbled words, interpret
  charitably. Hebrew/English mix is fine.

### Working style
- Wants direct engineering collaboration, not hand-holding — implement what's asked,
  explain non-obvious decisions inline or in commit messages, don't lecture.
- Tests ideas before implementing — good instinct, reinforce it.
- Asks sharp architectural questions — answer directly and completely.
- Confirm before destructive/hard-to-reverse actions (deleting files, force-push,
  rewriting core docs) unless already explicitly told to proceed.

---

## Current status

| Area | Status |
|---|---|
| Ingestion (loaders, 3 chunking strategies) | ✅ Done, tested |
| Dual indexing (ChromaDB + BM25) | ✅ Done, idempotent upserts |
| Hybrid retrieval (RRF + cross-encoder rerank) | ✅ Done, graceful degradation on rerank failure |
| Generation (citations + LLM-as-judge + confidence) | ✅ Done |
| FastAPI backend (`/ask`, `/ingest`, `/ingest/upload`, `/documents`, `/health`) | ✅ Done |
| React dashboard (ask, compare modes, drag-and-drop ingest) | ✅ Done |
| Docker (backend + frontend + compose) | ✅ Done |
| RAGAS eval harness + 30-question golden dataset | ✅ Done — see numbers below |
| Portfolio-facing README with architecture + numbers | 🔄 this file's sibling, `README.md` |
| MCP server (`retrieve` tool, raw retrieval, stdio) | ✅ Done — `src/mcp_server/server.py`, `/retrieve` route added alongside it |
| Demo video (≤4 min) | ⬜ Not started |

### Latest eval numbers (`evals/results/20260728T195847Z.json`, 30 questions)

| Metric | Score |
|---|---|
| Faithfulness | 0.87 |
| Answer relevancy | 0.85 |
| Context precision | 0.23 |
| Context recall | 0.42 |

Generation quality is solid; retrieval precision/recall are the known weak point and
the active investigation — see `evals/ragas/diagnose_precision.py` (compares golden
`reference_contexts` against actually-retrieved contexts per question, resolves each
snippet back to its source file) and `future/README.md` before starting new retrieval
work.

---

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

## File map

```
src/
├── config.py               ← Settings (pydantic-settings), ChunkingStrategy enum. Validates at import.
├── ingestion/
│   ├── base.py             ← RawDocument, Chunk, DocumentMetadata, Source dataclasses; interfaces
│   ├── loader.py           ← load_file()/load_directory(), FileExtension enum (factory dispatch)
│   ├── loaders/            ← PDFLoader, MarkdownLoader, HTMLLoader, TextLoader
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
│   └── routes.py               ← all endpoint handlers, incl. /retrieve (raw retrieval, no generation)
└── mcp_server/
    └── server.py            ← FastMCP server, `retrieve` tool, own Embedder/Reranker built in-process
                                 (stdio subprocess — doesn't depend on the FastAPI app running)

evals/
├── ragas/
│   ├── golden_dataset.json    ← 30 hand-written Q&A pairs (NEVER LLM-generated — see below)
│   ├── run_eval.py            ← RAGAS harness, saves timestamped results to evals/results/
│   ├── diagnose_precision.py  ← post-hoc precision/recall failure analysis, no new API calls
│   ├── audit_golden.py, inspect_question.py ← other diagnostic tools
│   └── _shared.py             ← shared ingest/golden-dataset loading
└── results/                   ← timestamped run outputs (.json + .jsonl)

frontend/                    ← React dashboard, see frontend/README.md
future/                      ← backlog of deferred ideas/hardening/bugs — check before starting new work
data/                        ← gitignored: data/chroma/ (vector index), data/uploads/ (ingested files)
tests/                       ← pytest unit tests; llm_eval-marked tests make real LLM calls (excluded by default)
```

---

## Key engineering conventions — enforce these by name

| Practice | What it means here |
|---|---|
| **Fail fast** | Config validates at import; missing `OPENAI_API_KEY` crashes loudly at startup, not silently mid-run |
| **Idempotency via path-derived IDs** | `Embedder.generate_id()` builds ChromaDB IDs from the file path. Anything that changes how files land on disk must preserve stable, repeatable paths or re-ingestion silently duplicates instead of upserting |
| **Carry metadata early** | Page, section, source — lost in the loader means lost forever downstream |
| **Graceful degradation** | `hybrid_retrieve()` falls back to plain RRF order if the cross-encoder reranker throws — a reranker failure must not take down `/ask` |
| **Batch API calls** | Embed in batches (`EMBEDDING_BATCH_SIZE`), never one chunk per HTTP call |
| **Temperature 0 for factual generation** | Deterministic output makes eval reproducible |
| **Golden eval data is human-only** | `golden_dataset.json` is never LLM-generated — defeats the purpose of the eval |
| **Lead docs with numbers** | README opens with eval scores, not setup instructions |
| **Per-stage metrics** | Retrieval and generation are measured separately (RAGAS breaks out faithfulness/relevancy vs. precision/recall) — you can't fix what you can't localize |
| **`/ask`'s citation contract** | `generate_answer()` numbers retrieved chunks `[1]`, `[2]`, ... (1-based, by position, not `doc_id`). Frontend citation-chip rendering depends on this — see `frontend/src/api/client.js` |
| **No auth, deliberately** | Single-user local/portfolio tool. Don't expose beyond localhost without addressing `future/README.md`'s hardening items first |

---

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env               # fill in OPENAI_API_KEY at minimum
uvicorn src.api.main:app --port 8000     # backend — do NOT add --reload on Windows here,
                                          # it's been observed serving a stale copy (future/README.md)

cd frontend && npm install && npm run dev   # dashboard, http://localhost:5173

pytest                              # unit tests (excludes llm_eval-marked tests)
python -m evals.ragas.run_eval      # full RAGAS eval run against the golden dataset

docker compose up --build           # both services containerized

mcp dev src/mcp_server/server.py    # MCP inspector — list/call the retrieve tool locally
```

There's also a CLI at `main.py` (insert documents, view the index, ask questions)
that talks to the retrieval/generation modules directly, bypassing the API.

---

## How to start each session

1. Check the status table above and `future/README.md` for open items
2. Ask what Rafael is working on if it's not obvious from context
3. Plan before large changes; implement directly for anything scoped and clear
