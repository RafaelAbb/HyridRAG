import { useState } from "react";
import { uploadFiles } from "../api/client.js";

const STRATEGIES = ["fixed", "recursive", "semantic"];

function withId(entry) {
  return { ...entry, id: crypto.randomUUID() };
}

// A dropped folder's DataTransferItemList only exposes the top-level
// FileSystemDirectoryEntry, not its nested contents — unlike webkitdirectory
// inputs, which the browser pre-flattens for you. This walk is the part
// that's required specifically to support folder drag-and-drop.
async function traverseEntry(entry, pathPrefix = "") {
  if (entry.isFile) {
    const file = await new Promise((res, rej) => entry.file(res, rej));
    return [{ file, relativePath: pathPrefix + entry.name }];
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    // readEntries() batches large directories rather than returning
    // everything at once — must be called repeatedly until it returns [].
    let batch;
    const entries = [];
    do {
      batch = await new Promise((res, rej) => reader.readEntries(res, rej));
      entries.push(...batch);
    } while (batch.length > 0);
    const nested = await Promise.all(
      entries.map((e) => traverseEntry(e, `${pathPrefix}${entry.name}/`))
    );
    return nested.flat();
  }
  return [];
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function IngestForm({ onIngested }) {
  const [stagedFiles, setStagedFiles] = useState([]);
  const [strategy, setStrategy] = useState("recursive");
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  function addFiles(newEntries) {
    setStagedFiles((prev) => [...prev, ...newEntries.map(withId)]);
  }

  function removeFile(id) {
    setStagedFiles((prev) => prev.filter((f) => f.id !== id));
  }

  function handlePickFiles(e) {
    const entries = Array.from(e.target.files).map((file) => ({ file, relativePath: file.name }));
    addFiles(entries);
    e.target.value = ""; // allow re-selecting the same file(s) later
  }

  function handlePickFolder(e) {
    const entries = Array.from(e.target.files).map((file) => ({
      file,
      relativePath: file.webkitRelativePath || file.name,
    }));
    addFiles(entries);
    e.target.value = "";
  }

  async function handleDrop(e) {
    e.preventDefault();
    setIsDragOver(false);
    const items = Array.from(e.dataTransfer.items);
    const entries = items
      .map((item) => item.webkitGetAsEntry?.())
      .filter(Boolean);
    const results = await Promise.all(entries.map((entry) => traverseEntry(entry)));
    addFiles(results.flat());
  }

  function handleDragOver(e) {
    e.preventDefault(); // required or the browser navigates to the dropped file instead of firing drop
    setIsDragOver(true);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (stagedFiles.length === 0) {
      setError("Select or drop at least one file.");
      setStatus("error");
      return;
    }
    setStatus("loading");
    setError(null);
    try {
      const data = await uploadFiles({
        files: stagedFiles.map((s) => s.file),
        relativePaths: stagedFiles.map((s) => s.relativePath),
        strategy,
      });
      setResult(data);
      setStatus("success");
      setStagedFiles([]);
      onIngested?.();
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <form className="ingest-form" onSubmit={handleSubmit}>
      <h3>Ingest documents</h3>

      <div
        className={isDragOver ? "dropzone dropzone-active" : "dropzone"}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setIsDragOver(false)}
      >
        <p className="muted">Drag and drop files or a folder here</p>
        <div className="dropzone-buttons">
          <label className="file-picker-label">
            Choose files
            <input type="file" multiple onChange={handlePickFiles} disabled={status === "loading"} hidden />
          </label>
          <label className="file-picker-label">
            Choose folder
            <input
              type="file"
              webkitdirectory=""
              multiple
              onChange={handlePickFolder}
              disabled={status === "loading"}
              hidden
            />
          </label>
        </div>
      </div>

      {stagedFiles.length > 0 && (
        <ul className="staged-file-list">
          {stagedFiles.map((s) => (
            <li key={s.id}>
              <span className="staged-file-path">{s.relativePath}</span>
              <span className="staged-file-size">{formatSize(s.file.size)}</span>
              <button
                type="button"
                className="staged-file-remove"
                onClick={() => removeFile(s.id)}
                disabled={status === "loading"}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}

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
          {result.skipped?.length > 0 &&
            ` Skipped ${result.skipped.length}: ${result.skipped.map((s) => `${s.filename} (${s.reason})`).join(", ")}`}
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
