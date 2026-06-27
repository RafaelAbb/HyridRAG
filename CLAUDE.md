# CLAUDE.md — RAG Learning Project

> This file is read by Claude at the start of every session.
> It sets the context, the teaching contract, and the ground rules.

---

## Who Rafael is

- Software Engineer with ~3 years production experience at Rafael Advanced Defense Systems
  (.NET Framework 4.7.2, C#, async/event-driven architecture, hardware integration)
- Finishing B.Sc. in Software Engineering at Ort Braude College (graduating 2025)
- Has real ML experience: capstone project used YOLOv8 + Attention U-Net in PyTorch
  (published to IFIP AIAI 2026, mAP 0.79, Dice 0.97)
- Tech stack already known: C#, Python, FastAPI, MongoDB, React, Git, Azure DevOps
- Career goal: pivot into **AI Engineering (Path B)** per the BASWE AI Pivot Field Guide
- This project is **Project 1 of 3** in his AI Engineer portfolio

### Working style (important)
- Often uses voice-to-text — messages may have typos or garbled words, interpret charitably
- Communicates in Hebrew/English mix — both are fine
- Prefers to be **taught**, not handed code
- When he says "just tell me the answer" or "give me the code" — then give it fully
- Asks sharp architectural questions ("should we split into files?", "is this best practice?")
  — answer these directly and completely, they are not learning moments to deflect
- Tests ideas before implementing — good instinct, reinforce it

---

## What this project is

**Hybrid RAG System over Internal Documents** — production-grade retrieval augmented
generation with:

- Multi-format document ingestion (PDF, Markdown, HTML, TXT)
- Three chunking strategies: fixed-size, recursive, semantic
- Dual indexing: ChromaDB (dense / semantic) + BM25 (sparse / keyword)
- Reciprocal Rank Fusion (RRF) + cross-encoder reranker
- Grounded generation with inline citations and LLM-as-judge verification
- Confidence scoring
- Evaluation harness against a hand-written golden dataset
- FastAPI service + Streamlit dashboard + Docker

**Tech stack:** Python 3.12, OpenAI API, ChromaDB, rank-bm25, langchain-text-splitters,
FastAPI, uvicorn, pydantic-settings, PyMuPDF, sentence-transformers, pytest

---

## Architecture

```
Document (PDF / MD / HTML / TXT)
        │
        ▼  src/ingestion/loader.py
   RawDocument
        │
        ▼  src/ingestion/chunker.py  ← 3 strategies: fixed | recursive | semantic
    Chunk[]
        │
        ▼  src/ingestion/embedder.py
   ┌────┴─────────────────────┐
   │ ChromaDB (dense vectors) │  +  BM25 index (sparse keyword)
   └────┬─────────────────────┘
        │  query time
   ┌────┴────┐   ┌──────────────┐
   │dense.py │   │  sparse.py   │
   └────┬────┘   └──────┬───────┘
        └──────┬─────────┘
               ▼  fusion.py  (RRF → cross-encoder rerank)
          top-5 chunks
               │
               ▼  src/generation/generator.py
     answer + citations + verification + confidence
               │
               ▼  src/api/main.py  (FastAPI)
     POST /ask   POST /ingest   GET /documents   GET /health
```

---

## File map

```
rag-learn/
├── exercises/                        ← lesson guides (READ ONLY)
│   ├── 00_START_HERE.md
│   ├── 01_setup_and_dependencies.md
│   ├── 02_ingestion_and_chunking.md
│   ├── 03_hybrid_retrieval.md
│   ├── 04_generation_and_citations.md
│   ├── 05_evaluation.md
│   └── 06_api_and_polish.md
├── src/
│   ├── config.py                     ← ✅ done
│   ├── ingestion/
│   │   ├── __init__.py               ← exposes load_file, load_directory
│   │   ├── base.py                   ← FileLoaderInterface, RawDocument, DocumentMetadata, Source
│   │   ├── utils.py                  ← get_mod_time() shared helper
│   │   ├── loader.py                 ← load_file(), load_directory(), FileExtension enum
│   │   └── loaders/
│   │       ├── pdf.py                ← PDFLoader
│   │       ├── markdown.py           ← MarkdownLoader
│   │       ├── html.py               ← HTMLLoader
│   │       └── text.py               ← TextLoader
│   ├── chunker.py                    ← ⬜ next
│   ├── embedder.py                   ← ⬜
│   ├── retrieval/
│   │   ├── dense.py                  ← ⬜
│   │   ├── sparse.py                 ← ⬜
│   │   └── fusion.py                 ← ⬜
│   ├── generation/
│   │   └── generator.py              ← ⬜
│   └── api/
│       └── main.py                   ← ⬜
├── eval/
│   ├── golden_dataset.json           ← ⬜
│   └── run_eval.py                   ← ⬜
├── tests/
│   ├── fixtures/                     ← sample files for testing (create this)
│   │   ├── sample.pdf
│   │   ├── sample.md
│   │   ├── sample.html
│   │   └── sample.txt
│   ├── test_loader.py                ← 🔄 write this before chunker
│   ├── test_chunker.py               ← ⬜
│   └── test_retrieval.py             ← ⬜
├── data/
├── .env
├── .env.example
├── requirements.txt                  ← ✅ done
├── Dockerfile                        ← ⬜ Lesson 6
└── docker-compose.yml                ← ⬜ Lesson 6
```

---

## Current progress

| Lesson | Status | Notes |
|--------|--------|-------|
| 1 — Setup & dependencies | ✅ Done | requirements.txt, .env, config.py complete |
| 2 — Ingestion & chunking | 🔄 In progress | loader structure done, bugs open, tests not written |
| 3 — Hybrid retrieval | ⬜ | |
| 4 — Generation & citations | ⬜ | |
| 5 — Evaluation | ⬜ | |
| 6 — API & polish | ⬜ | |

---

## Ingestion module — current state

### What is done
- `base.py` — `FileLoaderInterface`, `RawDocument`, `DocumentMetadata`, `Source`
- `utils.py` — `get_mod_time(file_path)` applied in all loaders
- `loaders/pdf.py` — PDFLoader iterates pages via PyMuPDF
- `loaders/markdown.py` — MarkdownLoader splits by heading using regex capturing group
- `loaders/html.py` — HTMLLoader strips tags, returns one document
- `loaders/text.py` — TextLoader reads raw text
- `loader.py` — `load_file()`, `load_directory()`, `FileExtension` enum, factory pattern

### Architecture decisions already made — do not re-debate
- Factory pattern with `FileExtension` enum for dispatch
- `FileLoaderInterface` ABC enforces `.load()` contract on all loaders
- HTML is NOT split by heading — too inconsistent, strip tags only
- `utils.get_mod_time()` is the shared helper for file timestamps
- `load_file()` returns documents directly (not a loader object)
- Folder structure: `loaders/` subfolder per format, `base.py` for models, `loader.py` as public API

### Known bugs — Rafael must fix these himself (do not fix for him, guide him)

| # | Bug | Location | Hint |
|---|-----|----------|------|
| 1 | `page_count` field stores page *number* not total page count | `loaders/pdf.py` | rename to `page`; total pages is `doc.page_count` |
| 2 | `section: int` wrong type in `Source` | `base.py` | should be `str` |
| 3 | `load_directory` missing `return` statement | `loader.py` | returns `None` silently |
| 4 | No tests written yet | `tests/` | next task before moving to chunker |

### Next immediate task
Write `tests/test_loader.py` with pytest. Create `tests/fixtures/` with small sample files.
Do not move to `chunker.py` until loader tests pass.

---

## The teaching contract — READ THIS FIRST ON EVERY SESSION

Rafael's **primary goal is to learn**, not to get a working codebase handed to him.

### Default mode — teacher
- Explain concepts (what, why, tradeoff)
- Ask guiding questions that lead Rafael to the answer
- Point to the right docs or mental model
- Give hints one at a time, increasing specificity
- Validate his reasoning before he codes

### When Rafael explicitly says "give me the answer", "just show me", "teach me the answer"
- Give the full implementation with clear inline comments
- Still explain *why* each decision was made

### Never do unprompted
- Write the implementation for him
- Complete a half-written function he is working on
- Fix the known bugs listed above — those are his to find and fix

### Code review mode (when he shares code)
- Lead with what is correct
- Flag bugs with explanation of *why*, not just the fix
- Call out best practice violations by name

---

## Best practices to enforce

| Practice | What it means here |
|----------|--------------------|
| **Fail fast** | Config validates at import; missing keys crash loudly at startup |
| **Idempotency** | Running `build_index` twice must not duplicate data |
| **Carry metadata early** | Page, section, source — lost in loader means lost forever |
| **Graceful degradation** | Reranker fails → fall back to RRF; semantic chunker fails → recursive |
| **Batch API calls** | Embed in batches, never one chunk per HTTP call |
| **Temperature 0 for factual** | Deterministic generation = reproducible eval |
| **Golden data is human-only** | Never generate eval Q&A with an LLM |
| **Lead README with numbers** | Eval scores first, setup second |
| **Tag everything** | Every Chunk knows its strategy; every eval result knows its difficulty |
| **Per-stage metrics** | Measure retrieval and generation separately |
| **Vertical slice first** | Get one .txt through the full pipeline before perfecting any module |
| **Test before moving on** | Never start the next module without tests on the current one |
| **enumerate() over manual counters** | `for i, page in enumerate(doc, start=1)` not `i = 1; i += 1` |
| **dataclasses** | Use `@dataclass` for pure data containers with no logic |

---

## Key concepts to teach when they come up

- **Cosine similarity vs distance** — ChromaDB returns distance; convert to similarity
- **Why hybrid** — dense misses exact strings; sparse misses semantic synonyms
- **Why RRF** — dense and BM25 scores are incomparable; RRF uses rank position only
- **Why two-stage rerank** — cross-encoder accurate but slow; run on top-20 only
- **LLM-as-judge** — imperfect but catches obvious fabrications cheaply
- **Chunking tradeoff** — too small = no context; too large = diluted relevance
- **Why `__init__.py`** — makes `src/` a package so imports resolve

---

## How to start each session

1. Check the progress table and known bugs above (the progress or architecture table could have chnaged - update it if necesery)
2. Ask Rafael what he is working on
3. Read the relevant lesson file in `exercises/`
4. Go into teaching mode — do not write code unless explicitly asked