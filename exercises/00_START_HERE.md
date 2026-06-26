# Hybrid RAG System — Build It Yourself

> **Project 6 of your AI Engineer portfolio.** You write every line. These lessons teach
> the *concepts* and *decisions*; they do not hand you the code. When you're stuck, that's
> the signal to experiment first, then ask me a specific question.

---

## How these lessons work

Each lesson maps to one file (or small group of files) in the skeleton. The teaching contract:

- **I explain the *what* and the *why*.** The concept, the design decision, the tradeoff.
- **You write the *how*.** The actual implementation is yours.
- **Checkpoints** let you self-verify before moving on.
- **"Ask me about…"** prompts tell you what's worth a conversation when you hit a wall.
- **Best-practice flags** (🛡️) mark habits that separate a portfolio piece from a tutorial repo.

Don't read all seven lessons up front. Do one, build it, run the checkpoint, then open the next.

---

## The architecture you're building

```
   ┌─────────────┐
   │  Document   │   PDF · Markdown · HTML · TXT
   └──────┬──────┘
          │  Lesson 2 ── loader.py
   ┌──────▼──────┐
   │ RawDocument │
   └──────┬──────┘
          │  Lesson 2 ── chunker.py  (fixed | recursive | semantic)
   ┌──────▼──────┐
   │   Chunk[]   │
   └──────┬──────┘
          │  Lesson 2 ── embedder.py
   ┌──────▼───────────────────────────┐
   │  ChromaDB (dense)  +  BM25 (sparse) │   two indexes, same chunks
   └──────┬───────────────────────────┘
          │  query
   ┌──────▼──────┐   ┌──────────────┐
   │ dense.py    │   │  sparse.py   │     Lesson 3
   └──────┬──────┘   └──────┬───────┘
          └────────┬────────┘
                   │  Reciprocal Rank Fusion → cross-encoder reranker  (fusion.py)
            ┌──────▼──────┐
            │  top-5 chunks │
            └──────┬──────┘
                   │  Lesson 4 ── generator.py
            ┌──────▼────────────────────────┐
            │ grounded answer + citations +  │
            │ verification + confidence score │
            └──────┬─────────────────────────┘
                   │  Lesson 6 ── api/main.py  (FastAPI)
            ┌──────▼──────┐
            │  /ask  /ingest  /documents │
            └─────────────┘

   Lesson 5 ── eval/  runs the whole thing against a golden dataset.
```

---

## Lesson order

| # | Lesson | Files you'll write | Doc phase |
|---|--------|--------------------|-----------|
| 1 | Setup & dependencies | `requirements.txt`, `.env`, `config.py` | Pre-work |
| 2 | Ingestion & chunking | `loader.py`, `chunker.py`, `embedder.py` | Phase 1 |
| 3 | Hybrid retrieval | `dense.py`, `sparse.py`, `fusion.py` | Phase 2 |
| 4 | Generation & citations | `generator.py` | Phase 3 |
| 5 | Evaluation | `golden_dataset.json`, `run_eval.py` | Phase 4 |
| 6 | API & polish | `api/main.py`, Docker, dashboard | Phase 5–6 |

---

## The one rule that matters most

**Build vertically, not horizontally.** Don't perfect `loader.py` before you've seen an
end-to-end answer come out the other side. Get the thinnest possible path working first
(load one `.txt` → one chunking strategy → dense-only retrieval → a generated answer),
*then* layer in sparse search, fusion, reranking, citation verification, and eval. A system
that runs end-to-end on day 3 teaches you more than three polished modules that never connect.

Start with **`01_setup_and_dependencies.md`**.
