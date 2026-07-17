import { useEffect, useState } from "react";
import { fetchDocuments } from "../api/client.js";

export function DocumentsPanel({ refreshKey }) {
  const [status, setStatus] = useState("idle");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  async function load() {
    setStatus("loading");
    setError(null);
    try {
      const result = await fetchDocuments();
      setData(result);
      setStatus("success");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  return (
    <div className="documents-panel">
      <div className="documents-panel-header">
        <h3>Indexed documents</h3>
        <button type="button" onClick={load} disabled={status === "loading"}>
          Refresh
        </button>
      </div>

      {status === "loading" && <p className="muted">Loading...</p>}
      {status === "error" && (
        <p className="error-alert" role="alert">
          Error: {error}
        </p>
      )}
      {status === "success" && data && data.sources.length === 0 && (
        <p className="empty-state">No documents indexed yet. Use the form above to ingest one.</p>
      )}
      {status === "success" && data && data.sources.length > 0 && (
        <>
          <p className="muted">{data.chunk_count} chunk(s) across {data.sources.length} document(s)</p>
          <ul>
            {data.sources.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
