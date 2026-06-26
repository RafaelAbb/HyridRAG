# Lesson 1 — Setup & Dependencies

**Goal:** a working Python environment, secrets handled safely, and a typed config object —
before you write a single line of RAG logic.

**Files you'll fill:** `requirements.txt`, `.env` (from `.env.example`), `src/config.py`

***

## 1.1 — Install the system tools

These live on your machine, not in the project. Install once.

| Tool               | Why you need it                                                              | Link                                              |
| ------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------- |
| **Python 3.12**    | Runtime. 3.12 is the ML-ecosystem sweet spot (3.14 is too new for some libs) | <https://www.python.org/downloads/>               |
| **Git**            | Version control — non-negotiable for a portfolio project                     | <https://git-scm.com/downloads>                   |
| **VS Code**        | Editor with great Python + Docker extensions                                 | <https://code.visualstudio.com/download>          |
| **Docker Desktop** | You'll containerise at the end (Phase 6)                                     | <https://www.docker.com/products/docker-desktop/> |

You'll also need an **OpenAI API key** (for embeddings + generation):
<https://platform.openai.com/api-keys> — fund it with \~\$5, this whole project costs cents.

**Checkpoint 1.1** — all four commands print a version:

```Shell
python --version    # 3.12.x   (might be `python3` on Mac/Linux)
git --version
docker --version
code --version
```

***

## 1.2 — Virtual environment

🛡️ **Best practice:** every project gets its own isolated environment. Never `pip install`
into your global Python — dependency versions from one project will silently break another.

The concept: a `venv` is a private copy of Python + packages that lives inside your project
folder. You "activate" it, install into it, and it can't pollute anything else.

You need to: create a venv, activate it, and confirm your shell prompt shows it's active.
The Python docs walk through the exact commands per OS:
<https://docs.python.org/3/library/venv.html>

**Ask me about…** the difference between `venv`, `conda`, `poetry`, and `uv` if you want to
know why I'm steering you to plain `venv` for this project.

**Checkpoint 1.2** — `which python` (Mac/Linux) or `where python` (Windows) points *inside*
your project folder, not to the system Python.

***

## 1.3 — `requirements.txt`

This file pins every Python dependency so the project is reproducible. Your job: list the
libraries this project needs, grouped by purpose with comments.

You don't need to guess versions from memory — the right habit is to install what you need,
then freeze. But for a portfolio project you want *intentional* pins, not a 200-line
`pip freeze` dump.

From the project spec, here's the **role** each dependency plays. Figure out the package
name for each (the PyPI search at <https://pypi.org> is your friend):

* An official client to call **OpenAI** (embeddings + chat)
* An official client to call **Anthropic** (optional, for a generation fallback)
* A **vector database** that persists to disk with zero server setup *(hint: the spec names ChromaDB)*
* A **BM25** implementation for keyword search *(hint:* *`rank_bm25`)*
* **Text splitters** for chunking *(hint: the standalone* *`langchain-text-splitters`, not all of LangChain)*
* A **web framework** that's async-native *(hint: FastAPI)* + the server that runs it *(hint: uvicorn)*
* **Data validation** with type enforcement *(hint: Pydantic + its settings extension)*
* A **PDF parser** *(hint: PyMuPDF — imported as* *`fitz`)*
* A **cross-encoder** for reranking *(hint: sentence-transformers)*
* A **test runner** *(hint: pytest)*

🛡️ **Best practice:** pin with `>=` minimum versions (e.g. `openai>=1.35.0`) so you get
bug fixes but document the floor you tested against.

**Ask me about…** why I split `langchain-text-splitters` out instead of installing the full
`langchain` meta-package. (It's a real production instinct worth understanding.)

**Checkpoint 1.3** — `pip install -r requirements.txt` completes with no errors, and
`pip list` shows everything.

***

## 1.4 — Secrets: `.env`

🛡️ **The single most important security habit in this project:** your API key never touches
your source code and never gets committed to Git.

The pattern, which you'll see in every professional Python repo:

* `.env` holds real secrets — and is listed in `.gitignore` so it's never committed.
* `.env.example` holds the *shape* (keys with placeholder values) — and *is* committed, so
  teammates know what to fill in.

Your job:

1. Fill `.env.example` with every config key this project needs, using fake values.
   Think about what's configurable: the embedding model name, ChromaDB paths, retrieval
   `top_k` values, the RRF weights, chunk size/overlap, the generation model.
2. Copy it to `.env` and put your real OpenAI key in.
3. Make sure `.gitignore` contains `.env` (and `.venv/`, `__pycache__/`, `data/chroma/`).

**Checkpoint 1.4** — `git status` does **not** list `.env` as a file to be committed. If it
does, your `.gitignore` is wrong — fix it before you ever commit.

***

## 1.5 — Typed config: `src/config.py`

Instead of scattering `os.getenv("OPENAI_API_KEY")` calls across the codebase, you load all
config *once* into a single typed object. This is where `pydantic-settings` earns its place.

The concept: you define a class describing every setting (with types and defaults), point it
at your `.env` file, and Pydantic validates and loads everything on import. If a required key
is missing or mistyped, you get a clear error at startup — not a mysterious `None` three
modules deep.

Read how `BaseSettings` works: <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>

Your job: write a `Settings` class that mirrors the keys in your `.env`, with sensible
defaults for the non-secret ones (e.g. `dense_top_k: int = 10`). Export a single
`settings = Settings()` instance that the rest of the code imports.

🛡️ **Best practice:** defaults in code, secrets in `.env`. Anyone reading `config.py` sees
the full set of tunable knobs in one place — that's documentation that can't go stale.

**Ask me about…** why a typed config object beats passing a `dict` of settings around, or
why "fail fast at startup" matters more than it sounds.

**Checkpoint 1.5** — open a Python shell, run `from src.config import settings`, and print
`settings.embedding_model`. It works and shows your value. You're ready for Lesson 2.

***

## What "done" looks like for Lesson 1

* [x] Four tools installed, all print versions
* [x] venv created and active
* [x] `requirements.txt` written and installs cleanly
* [x] `.env` holds your key and is git-ignored (`.env.example` is committed)
* [x] `config.py` loads and validates on import
* [x] First commit made (`git init`, `git add`, `git commit`) — with `.env` absent from it

Next: **`02_ingestion_and_chunking.md`**
