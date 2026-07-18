const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

if (import.meta.env.PROD && !import.meta.env.VITE_API_BASE_URL) {
  // A missing env var here silently bakes "localhost:8000" into the built
  // bundle — every deployed user's browser would try to reach their own
  // machine and fail with no explanation. Dev fallback, prod footgun.
  console.error(
    "[rag-dashboard] VITE_API_BASE_URL is not set in this production build — " +
      "API calls will target localhost and fail for deployed users. Set VITE_API_BASE_URL at build time."
  );
}

// AskResponse.answer contains inline "[n]" markers (1-based) referring to
// sources[n-1] — generate_answer's context_block is built via
// enumerate(retrieved_results, start=1) and the model is prompted to cite
// that number. AnswerPanel relies on this to render clickable citations.
export async function askQuestion({ question, topK, retrievalMode, signal }) {
  return request("/ask", "POST", {
    question,
    top_k: topK,
    retrieval_mode: retrievalMode, // "hybrid" | "dense" | "sparse"
  }, signal);
}

export async function ingestPath({ path, strategy }) {
  return request("/ingest", "POST", { path, strategy }); // strategy: "fixed" | "recursive" | "semantic"
}

// Bypasses request()'s JSON-only helper on purpose — request() exists so
// every other caller gets "always JSON" for free, and there's exactly one
// multipart caller in the app. Branching request() for this one case would
// blur its contract for zero reuse benefit.
export async function uploadFiles({ files, relativePaths, strategy, signal }) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  relativePaths.forEach((p) => formData.append("relative_paths", p));
  formData.append("strategy", strategy);

  const res = await fetch(`${BASE_URL}/ingest/upload`, {
    method: "POST",
    // Do NOT set Content-Type here — the browser computes the multipart
    // boundary itself; setting this header manually breaks the boundary
    // parameter and the server can't parse the body at all.
    body: formData,
    signal,
  });
  if (!res.ok) throw await toApiError(res);
  return res.json();
}

export async function fetchDocuments() {
  return request("/documents", "GET");
}

export async function fetchHealth() {
  return request("/health", "GET");
}

async function request(path, method, body, signal) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!res.ok) throw await toApiError(res);
  return res.json();
}

async function toApiError(res) {
  const body = await res.json().catch(() => null);
  // FastAPI's HTTPException(detail="...") -> { detail: "<string>" }
  // Pydantic validation errors (422) -> { detail: [{ loc, msg, type }, ...] }
  const detail = body?.detail;
  const message = Array.isArray(detail)
    ? detail.map((e) => e.msg).join("; ")
    : detail ?? res.statusText;
  return new Error(message);
}
