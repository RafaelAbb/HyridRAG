# RAG Dashboard — Frontend

React (Vite) dashboard for the RAG backend one directory up. Ask questions against the indexed
documents with a choice of retrieval strategy, or run a side-by-side dev comparison; ingest new
documents via drag-and-drop. Written for whoever (human or AI agent) picks this up next.

## Stack and philosophy

Plain React + Vite, **no other dependencies** — no router, no CSS framework, no data-fetching
library, no dropzone library. Every "should we add a dependency for X" question in this codebase
was answered "no" deliberately (see inline comments at the relevant call sites) to keep the app
small and its behavior traceable. If you're tempted to add one, that's a real decision — leave a
comment explaining why the tradeoff changed, don't just add it quietly.

## Running it

```bash
npm install
cp .env.example .env    # VITE_API_BASE_URL, defaults to http://localhost:8000
npm run dev              # http://localhost:5173
```

Requires the backend running (`uvicorn src.api.main:app --port 8000` from the repo root) —
`/health` is polled on load and a banner shows if it's not reachable yet.

`npm run build` produces a static bundle; `npm run preview` serves it locally. If
`VITE_API_BASE_URL` isn't set at build time in a production build, `api/client.js` logs a
console warning rather than failing silently (see `future/README.md` for the stronger
build-time-hard-fail option that was considered and deferred).

## Folder map

```
src/
├── main.jsx                     ← entrypoint, wraps <App/> in <ErrorBoundary/>
├── App.jsx                       ← tabs (Ask / Manage Index), lifted form state, question history wiring
├── styles.css                     ← the only stylesheet — design tokens (:root) + all component styles
├── api/
│   └── client.js                   ← one function per endpoint; askQuestion/uploadFiles/ingestPath/fetchDocuments/fetchHealth
├── hooks/
│   ├── useAsk.js                    ← per-retrieval-mode request state: status/data/error/tookMs, AbortController-based cancellation
│   └── useQuestionHistory.js         ← localStorage-backed question history
└── components/
    ├── QuestionForm.jsx                ← question input + top_k + retrieval-mode selector (hybrid/dense/sparse/compare)
    ├── SingleAnswerView.jsx             ← one useAsk() instance for a single chosen strategy
    ├── ComparisonView.jsx                ← 3 fixed useAsk() instances (hybrid+dense+sparse), the "dev" comparison mode
    ├── AnswerPanel.jsx                    ← renders one answer: citations, confidence, latency, loading skeleton
    ├── ConfidenceMeter.jsx                  ← confidence bar
    ├── ChunkList.jsx                         ← ranked source chunks with citation-click highlighting
    ├── QuestionHistory.jsx                    ← <details> disclosure, click to re-run a past question
    ├── IngestForm.jsx                           ← dropzone: click multi-file, click folder, or drag-and-drop
    ├── DocumentsPanel.jsx                        ← lists indexed sources + chunk count
    └── ErrorBoundary.jsx                          ← app-level React error boundary
```

## Things to know before changing this code

- **Retrieval modes**: `hybrid` (default), `dense`, `sparse` each render one `AnswerPanel` via
  `SingleAnswerView`. `compare` runs all three in parallel via `ComparisonView` and is framed in
  the UI as a secondary/dev-oriented mode (visually separated with a divider + "dev" tag in
  `QuestionForm`), not the primary flow.
- **Citation numbering contract**: `AskResponse.answer` contains inline `[n]` markers, 1-based,
  mapping to `sources[n-1]` — this comes from how the backend's prompt numbers retrieved chunks
  (see the comment in `api/client.js`). `AnswerPanel` depends on this exact convention to render
  clickable citation chips.
- **Request cancellation**: `useAsk` aborts a hook instance's own in-flight request whenever a
  new one starts on the same instance (per retrieval mode) — so firing a new question doesn't
  race a stale one. Aborted requests are silently dropped (`err.name === "AbortError"`), not
  surfaced as errors.
- **`IngestForm`'s upload path**: three ways files get staged (click-multi-file, click-folder via
  a *separate* `webkitdirectory` input, or drag-and-drop with recursive `FileSystemEntry`
  traversal) all funnel into the same `relativePath`-tagged staging list, then upload via
  `uploadFiles()` in `api/client.js` (FormData, deliberately not routed through the shared JSON
  `request()` helper — see the comment there).
- **Tab switching uses `display:none`, not conditional unmount** (in `App.jsx`) — switching
  between Ask and Manage Index used to silently re-fire the last question on return; both
  sections now stay mounted so in-flight/completed request state survives a tab switch.
- **No upload-progress bar, no per-panel error boundaries, no build-time env-var hard-fail** —
  all deliberate scope calls, documented with reasoning in `../future/README.md`.

## Known environment constraint

Whoever developed the backend side of recent features did so on a machine with no Node.js
installed and could not run `npm install`/`npm run dev` themselves — frontend changes from that
context were reviewed statically (syntax, import resolution, prop-shape matching) but need a
real browser pass, especially anything touching drag-and-drop/`webkitdirectory` (browser-specific
behavior that can't be verified without an actual browser).
