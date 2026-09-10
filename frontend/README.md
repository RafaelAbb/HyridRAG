# RAG Dashboard — Frontend

React (Vite) dashboard for the RAG backend one directory up. Ask questions against the indexed
documents with a choice of retrieval strategy, or run a side-by-side comparison; ingest new
documents via drag-and-drop.

Plain React + Vite, no other runtime dependencies (no router, no CSS framework, no data-fetching
library).

## Running it

```bash
npm install
cp .env.example .env    # VITE_API_BASE_URL, defaults to http://localhost:8000
npm run dev              # http://localhost:5173
```

Requires the backend running (`uvicorn src.api.main:app --port 8000` from the repo root) —
`/health` is polled on load and a banner shows if it's not reachable yet.

`npm run build` produces a static bundle; `npm run preview` serves it locally.

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
    ├── ComparisonView.jsx                ← 3 fixed useAsk() instances (hybrid+dense+sparse) side by side
    ├── AnswerPanel.jsx                    ← renders one answer: citations, confidence, latency, loading skeleton
    ├── ConfidenceMeter.jsx                  ← confidence bar
    ├── ChunkList.jsx                         ← ranked source chunks with citation-click highlighting
    ├── QuestionHistory.jsx                    ← <details> disclosure, click to re-run a past question
    ├── IngestForm.jsx                           ← dropzone: click multi-file, click folder, or drag-and-drop
    ├── DocumentsPanel.jsx                        ← lists indexed sources + chunk count
    └── ErrorBoundary.jsx                          ← app-level React error boundary
```
