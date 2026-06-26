# Lesson 6 — API & Portfolio Polish

**Goal:** wrap the pipeline in a real API, containerise it, and package it so a reviewer can
run it in one command and a recruiter can grasp it in 60 seconds.

**Files:** `src/api/main.py`, `Dockerfile`, `docker-compose.yml`, `README.md`, a dashboard
**Doc phase:** Phase 5–6 (Day 11–14)

The work is done; this lesson is what makes the work *count*. A brilliant pipeline that only
runs in your notebook is invisible. A deployed, documented, demoed system is a portfolio.

---

## 6.1 — `api/main.py`: the FastAPI service

This is where your backend strength is a genuine edge over data-scientist candidates — you
already think in clean APIs. Lean into it.

Endpoints the spec calls for:
- `POST /ask` — question in, answer + citations + confidence + sources out
- `POST /ingest` — index a new document
- `GET /documents` — list what's indexed
- `GET /health` — liveness (🛡️ every production service has one)

🛡️ **Best practices to apply here:**
- **Pydantic request/response models.** Type every endpoint's input and output. FastAPI turns
  these into automatic OpenAPI docs at `/docs` — free, interactive, and it impresses reviewers.
- **Right status codes.** 404 when a file isn't found, 503 when no index is loaded yet, 200 on
  success. Sloppy status codes read as junior.
- **Load the index once at startup, not per request.** Use FastAPI's lifespan handler. Loading
  ChromaDB + the BM25 index on every `/ask` would be painfully slow.

🧠 **Think about:** the cross-encoder reranker loads a model into memory. Where should that
happen — per request, or once at startup? What does that mean for your container's memory and
cold-start time? (No single right answer; reason about the tradeoff and note it.)

FastAPI's first-steps tutorial is excellent: https://fastapi.tiangolo.com/tutorial/

**Checkpoint 6.1** — `uvicorn src.api.main:app --reload`, open `http://localhost:8000/docs`,
and drive the whole pipeline from the interactive UI. Ingest a doc, ask a question, see the
cited answer.

---

## 6.2 — A query dashboard

A simple UI turns an abstract API into something a non-technical reviewer *feels*. The spec
suggests Streamlit (fastest path) or React (if you want to show frontend range — you have
React on your resume, so this is a low-cost flex).

What it should show:
- a question box
- the generated answer with clickable/visible citations
- the retrieved chunks ranked by score
- the confidence breakdown
- 🧠 **the money feature:** a toggle for **hybrid vs dense-only** retrieval, side by side. This
  makes your Lesson 3 work *visible* — the reviewer sees hybrid win in real time. That's the
  most persuasive thing in the whole project.

Streamlit docs: https://docs.streamlit.io/

**Checkpoint 6.2** — a non-developer could open the dashboard, ask a question, and understand
what hybrid retrieval bought them.

---

## 6.3 — Containerise

🛡️ **Why this matters for *you* specifically:** the field guide is emphatic that your
production-engineering instincts are your edge in the pivot. A clean Dockerfile +
docker-compose is you *demonstrating* that edge, not just claiming it.

- **Dockerfile** — package the app. 🧠 Think about: what system libraries does PyMuPDF need?
  How do you avoid re-installing dependencies on every code change (layer caching — copy
  `requirements.txt` and install *before* copying your source)?
- **docker-compose.yml** — orchestrate the service + persist the `data/` volume so the index
  survives container restarts.
- 🛡️ **A seed script** so a reviewer runs `docker compose up`, and sample docs are already
  indexed — they can ask questions in 30 seconds without setup. Removing friction for the
  reviewer is a senior instinct.

Docker's Python guide: https://docs.docker.com/language/python/

**Checkpoint 6.3** — `docker compose up` on a clean machine brings up a working, pre-seeded
system. No "works on my machine."

---

## 6.4 — The README and the demo

🧠 **Write the README like internal onboarding docs, not a tutorial.** A teammate joining your
team should be able to run it, understand the architecture, and know the design decisions.
Structure: one-line what-it-is, the eval numbers up top (lead with results), setup, usage,
architecture diagram, and a "design decisions" section explaining *why* (hybrid over dense,
which chunking won and the data, the citation-verification choice).

🛡️ **Lead with numbers.** "X% faithfulness, Y% citation accuracy, hybrid beat dense-only by Z
points on a 50-question suite." Recruiters skim; the numbers are the hook.

**The demo (≤4 min Loom):** ingest docs → ask questions of rising difficulty → show the
citation verifier catching a hallucination → flip the hybrid-vs-dense toggle. 🧠 The field guide
is right that a recorded walkthrough is more persuasive than any README — a reviewer watches
the thing *work*.

**Ask me about…** structuring the "design decisions" section, or what to say in the Loom to
frame this for a hiring manager.

**Checkpoint 6.4** — README leads with eval numbers and has an architecture diagram; a ≤4 min
demo video exists; the repo is public on GitHub with clean commit history.

---

## What "done" looks like for Lesson 6 — and the whole project

- [ ] FastAPI service, typed models, correct status codes, startup-loaded index, `/docs` works
- [ ] Dashboard with the hybrid-vs-dense toggle
- [ ] Dockerfile + compose + seed script — one-command spin-up
- [ ] README leading with numbers + architecture diagram + design-decisions section
- [ ] ≤4 min demo video
- [ ] Public GitHub repo, clean history

---

## You're now holding Project 1 of 3

This is your "production RAG system" — the first pillar of the AI Engineer portfolio. The
patterns you just internalised (embeddings, vector search, LLM-as-judge, eval discipline,
graceful degradation) feed *directly* into Project 15 (your agent / bureaucratic assistant —
its memory system reuses this exact vector-store work) and Project 10 (fine-tuning — same eval
discipline).

When you're ready, ask me to set up the Project 15 skeleton the same way.
