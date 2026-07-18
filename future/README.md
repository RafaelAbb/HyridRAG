# Future work

Backlog of ideas, hardening items, and known gaps surfaced during development but deliberately
deferred — not urgent, not forgotten. Add to this file (or split into topic files here) as new
items come up instead of letting them live only in chat history.

Each item: what it is, why it was deferred, and rough size if known.

---

## Backend

- **Harden or remove `/ingest`'s arbitrary server-side path.** Flagged twice by automated
  security review: `POST /ingest` takes a client-supplied filesystem path with no restriction on
  where it can point. Currently accepted as a deliberate tradeoff (single-user local tool, not
  multi-tenant), but if this API is ever exposed beyond localhost, either restrict `path` to an
  allowlisted base directory (`os.path.realpath` + prefix check) or remove the endpoint entirely
  in favor of `/ingest/upload` only. Also has no authentication at all — same caveat.
- **No authentication anywhere on the API.** Same "local tool" tradeoff as above. Would need an
  API-key or session dependency before this could safely run on a shared network.
- **`took_ms` field on `AskResponse`.** Latency is currently measured client-side
  (`performance.now()` around the fetch) — network + server round trip, not a true server-side
  breakdown (retrieval vs. generation vs. rerank time). Would need a field added to
  `AskResponse` and populated in `routes.py`'s `/ask` handler.
- **True confidence breakdown.** `AskResponse.confidence` is a single float. Splitting it into
  `retrieval_score` vs `citation_coverage` (both already computed internally by
  `calculated_confidence` in `src/generation/generator.py`) would need those exposed on
  `GenerationResult` (`src/generation/base.py`) and added to `AskResponse`.
- **Orphaned `data/uploads/` cleanup.** No expiry/cleanup mechanism for uploaded files — they
  accumulate forever. Fine at current scale; would need a sweep job or manual cleanup convention
  if this sees heavier use.
- **`Reranker.rerank()` returns the pre-rerank RRF score, not the cross-encoder score.** In
  `src/retrieval/fusion.py`, reranked results keep their original `.score` from the RRF merge —
  the cross-encoder's own score is computed but discarded. Minor, but means the `score` field in
  `/ask` responses for hybrid mode doesn't reflect what actually determined the final ranking.

## Frontend

- **Upload progress bar.** `fetch()` has no upload-progress event (only `XMLHttpRequest` does).
  Deliberately skipped — not worth maintaining two HTTP idioms for one call at this app's scale
  (see `IngestForm.jsx`'s dropzone comment). Revisit if uploads get large enough that "it's
  loading" stops being good enough feedback.
- **Per-panel error boundaries in Compare mode.** Currently one `ErrorBoundary` wraps the whole
  app (`main.jsx`). A malformed response in one of the 3 compare panels currently would take
  down the whole page rather than just that panel. Fine for now, worth revisiting if compare
  mode becomes more central to the demo.
- **Build-time hard-fail for missing `VITE_API_BASE_URL` in production.** Currently just a
  `console.error` warning at runtime (`client.js`). A `vite.config.js`-level check that fails
  the build outright would catch a misconfigured deploy earlier — deferred pending a decision on
  actual deploy target (Vercel/Netlify/etc. env var workflow).

## Project / lesson 6 checklist (not started)

- Dockerfile + docker-compose (lesson 6.3) — containerize both services, seed script for a
  one-command demo spin-up.
- Project README leading with eval numbers + architecture diagram + design-decisions section
  (lesson 6.4) — distinct from the dev-facing `README.md`/`frontend/README.md`, this is the
  portfolio-facing one.
- ≤4 min demo video (lesson 6.4).

## Environment gotchas worth remembering

- `uvicorn --reload` on this dev machine spawns its worker under the *global* Python install
  instead of the venv (visible via `multiprocessing.spawn` in the process list), which appeared
  to serve a stale/shadowed copy of the app after edits. Workaround: run without `--reload` and
  restart manually after backend changes. Worth root-causing properly if it becomes annoying
  (likely a `pyvenv.cfg`/`sys.executable` resolution issue).
