# Lesson 2 — Ingestion & Chunking

**Goal:** turn raw documents into clean, embedded, searchable chunks sitting in two indexes.

**Files:** `src/ingestion/loader.py`, `chunker.py`, `embedder.py`
**Doc phase:** Phase 1 (Day 1–3)

This is the foundation. Garbage in here means garbage retrieval forever — no clever reranking
saves a bad chunking strategy. Spend the care here.

***

## 2.1 — `loader.py`: documents → clean text

**The problem:** a PDF, a Markdown file, and an HTML page are three completely different
beasts on disk, but your pipeline wants one clean, uniform thing: text plus metadata about
where it came from.

**The design decision:** define one output type — call it `RawDocument` — that every loader
produces regardless of input format. A dataclass with `text`, `source` (file path), and
optional `page` / `section` is a clean start. This is the *adapter pattern*: messy inputs,
one tidy interface.

🛡️ **Best practice:** carry metadata from the very first step. When your RAG system later
cites "page 14 of contract.pdf," that page number had to survive all the way from this
loader. If you drop it here, you can't invent it later.

**What each format needs:**

* **PDF** — try native text extraction first (PyMuPDF / `fitz` is fast and free). Read the
  quickstart: <https://pymupdf.readthedocs.io/en/latest/the-basics.html>
  * 🧠 **Think about:** what do you do with a *scanned* PDF that has no extractable text? You
    don't have to solve OCR today — but leave a clear `TODO` and a detection check (hint: if
    a page yields almost no characters, it's probably an image). Naming the gap is senior
    behaviour; pretending it doesn't exist is not.
* **Markdown** — don't dump the whole file as one blob. Split it by heading so each section
  keeps its `section` metadata. A regex on `^#{1,3}\s+` lines gets you most of the way.
* **HTML** — strip the tags, collapse whitespace. (For this project a regex is fine; in
  production you'd reach for something like BeautifulSoup.)
* **TXT** — the easy one.

Write a single `load_file(path)` entry point that detects the extension and dispatches to the
right private loader. Add a `load_directory()` that walks a folder.

**Ask me about…** why structure-aware loading (keeping headings) matters so much for
retrieval quality later. It connects directly to chunking.

**Checkpoint 2.1** — point `load_file` at a real PDF and a real `.md`, print the results. You
get clean text with correct `source` and `page`/`section` populated.

***

## 2.2 — `chunker.py`: text → chunks (the heart of Phase 1)

**Why chunk at all?** Two hard constraints. (1) Embedding models have a token limit. (2) More
importantly — retrieval precision. If you embed a whole 40-page document as one vector, a
query matches the *average* of 40 pages, which is mush. Small, focused chunks mean the
retrieved context is actually about the question.

**The tradeoff you're navigating:** chunks too *small* lose context (a sentence with no
surrounding meaning); chunks too *large* dilute relevance and waste context-window space. This
tension is why the spec asks you to build **three strategies** and *measure* which wins
(that measurement happens in Lesson 5 — this is the setup for it).

🛡️ **Best practice:** tag every chunk with which strategy produced it. You can't compare
strategies in your eval later if you can't tell them apart.

**The three strategies:**

1. **Fixed-size + overlap** — slide a window of N characters with M overlap. Dead simple,
   your baseline. The *overlap* matters: without it, a sentence split across a chunk boundary
   is lost to both chunks. 🧠 Why does overlap help retrieval? Reason it through before you
   build it.

2. **Recursive** — split on natural boundaries in priority order: paragraphs, then sentences,
   then words, only falling back to hard character cuts when needed. You don't have to write
   the splitting algorithm from scratch — `langchain-text-splitters` has
   `RecursiveCharacterTextSplitter`. Read its docs and understand the `separators` list:
   <https://python.langchain.com/docs/concepts/text_splitters/>

3. **Semantic** — the interesting one. Split where the *topic* changes. Embed each sentence,
   walk through them, and start a new chunk when cosine similarity between adjacent sentences
   drops below a threshold. 🧠 **Think about:** this calls the embedding API at chunk time, so
   it's slower and costs tokens. When is that worth it? What's your fallback if the API call
   fails mid-run? (A robust system degrades to recursive splitting rather than crashing.)

Design a `Chunk` dataclass (text + source + chunk\_index + strategy + provenance) and a single
`chunk_documents(docs, strategy=...)` function that dispatches to the three.

**Ask me about…** cosine similarity itself if the math is fuzzy — it's the single most
important number in this entire project and it shows up again in retrieval and dedup.

**Checkpoint 2.2** — run all three strategies on the same document. Eyeball the output: do
the recursive chunks break on clean boundaries? Do the semantic chunks group related
sentences? Print chunk counts per strategy — they should differ.

***

## 2.3 — `embedder.py`: chunks → two indexes

This file does three jobs: embed, deduplicate, and write to **both** a dense and a sparse
index. They must stay in sync — every chunk lives in both.

**Embeddings, the concept:** an embedding turns text into a list of \~1500 numbers (a vector)
positioned so that similar meanings sit close together in space. "How do I reset my password"
and "I forgot my login" land near each other even with zero shared words. That's what dense
retrieval exploits. You'll use OpenAI's `text-embedding-3-small`. API reference:
<https://platform.openai.com/docs/guides/embeddings>

🛡️ **Best practice — batching:** don't call the embedding API once per chunk in a loop. Send
chunks in batches (the API accepts a list). This is faster and cheaper. Your config already
has an `embedding_batch_size` knob — use it.

**The two indexes:**

* **Dense** → ChromaDB. It stores vectors and does nearest-neighbour search. Set it up as a
  *persistent* client so your index survives a restart (you don't want to re-embed every
  launch — that's real money). Use cosine space. Quickstart:
  <https://docs.trychroma.com/docs/overview/getting-started>
* **Sparse** → BM25 (`rank_bm25`). This is classic keyword search — no embeddings, it scores
  documents by term overlap. 🧠 **Why keep both?** Embeddings miss exact strings: a function
  name like `LockBits`, an error code, an acronym. BM25 nails those. You'll fuse them in
  Lesson 3. This dense+sparse combo is *the* thing that makes it "hybrid" and the reason this
  project beats a tutorial RAG.

**Deduplication:** before inserting a chunk, check whether a near-identical one already
exists (cosine similarity > \~0.95). 🧠 **Why?** The same boilerplate paragraph might appear in
ten documents. Without dedup, your retriever wastes all five precious context slots on the
same text. Give each chunk a *stable ID* (hash of source + index + strategy) so re-running
ingestion is idempotent — it skips what's already indexed instead of duplicating it.

🛡️ **Best practice — idempotency:** running `build_index` twice should not double your data.
This is the difference between a script and a system.

**Ask me about…** how ChromaDB returns *distance* vs *similarity* (they're inverses and it's
a classic off-by-confusion bug), or why I'd persist the BM25 index to disk too.

**Checkpoint 2.3** — ingest a few documents. Then ingest the *same* documents again — your log
should say "0 new chunks (skipping N duplicates)." Query ChromaDB's collection count and
confirm it matches your chunk count. The index persists across a Python restart.

***

## What "done" looks like for Lesson 2

* [x] `loader.py` handles PDF/MD/HTML/TXT into a uniform `RawDocument`, metadata preserved
* [x] `chunker.py` implements all three strategies, each chunk tagged with its strategy
* [x] `embedder.py` embeds in batches, writes to ChromaDB + BM25, dedups, is idempotent
* [ ] You can go from a folder of docs to a populated, persistent index
* [x] A test or two in `tests/test_chunker.py` (🛡️ start the testing habit now)

**The vertical-slice milestone:** by end of Lesson 2 you have searchable data. Lesson 3 makes
it retrievable well.

Next: **`03_hybrid_retrieval.md`**
