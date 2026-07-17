import { useState } from "react";
import { ingestPath } from "../api/client.js";

const STRATEGIES = ["fixed", "recursive", "semantic"];

// Can't validate the path actually exists from the browser — that's
// inherently server-only. This just catches obviously-malformed input a
// beat earlier, with a clearer message than a raw 404 body.
function validatePath(path) {
  const trimmed = path.trim();
  if (!trimmed) return "Path is required.";
  if (trimmed.length > 1024) return "Path is too long.";
  if (/[\x00-\x1f]/.test(trimmed)) return "Path contains invalid control characters.";
  return null;
}

export function IngestForm({ onIngested }) {
  const [path, setPath] = useState("");
  const [strategy, setStrategy] = useState("recursive");
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    const validationError = validatePath(path);
    if (validationError) {
      setError(validationError);
      setStatus("error");
      return;
    }
    setStatus("loading");
    setError(null);
    try {
      const data = await ingestPath({ path: path.trim(), strategy });
      setResult(data);
      setStatus("success");
      onIngested?.();
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <form className="ingest-form" onSubmit={handleSubmit}>
      <h3>Ingest a document</h3>
      <input
        type="text"
        placeholder="Path to a file or folder on the server"
        value={path}
        onChange={(e) => setPath(e.target.value)}
        disabled={status === "loading"}
      />
      <select value={strategy} onChange={(e) => setStrategy(e.target.value)} disabled={status === "loading"}>
        {STRATEGIES.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      <button type="submit" disabled={status === "loading"}>
        Ingest
      </button>

      {status === "success" && result && (
        <p className="ingest-success">
          Ingested {result.documents_ingested} document(s) → {result.chunks_created} chunk(s)
          ({result.strategy}).
        </p>
      )}
      {status === "error" && (
        <p className="error-alert" role="alert">
          Error: {error}
        </p>
      )}
    </form>
  );
}
