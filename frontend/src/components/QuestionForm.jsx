const MODES = [
  { value: "hybrid", label: "Hybrid" },
  { value: "dense", label: "Dense" },
  { value: "sparse", label: "Sparse" },
];

function clampTopK(value) {
  if (value === "") return "";
  const n = Number(value);
  if (Number.isNaN(n)) return null; // signal "ignore this keystroke"
  return Math.min(50, Math.max(1, n));
}

export function QuestionForm({ question, topK, mode, onQuestionChange, onTopKChange, onModeChange, onSubmit, disabled }) {
  function handleTopKChange(e) {
    const clamped = clampTopK(e.target.value);
    if (clamped === null) return;
    onTopKChange(clamped);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!question.trim()) return;
    const safeTopK = topK === "" || Number.isNaN(Number(topK)) ? 5 : Math.min(50, Math.max(1, Number(topK)));
    onSubmit(question.trim(), safeTopK, mode);
  }

  return (
    <form className="question-form" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Ask a question about the indexed documents..."
        value={question}
        onChange={(e) => onQuestionChange(e.target.value)}
        disabled={disabled}
      />

      <div className="mode-select" role="radiogroup" aria-label="Retrieval mode">
        {MODES.map((m) => (
          <button
            key={m.value}
            type="button"
            role="radio"
            aria-checked={mode === m.value}
            className={mode === m.value ? "mode-btn active" : "mode-btn"}
            onClick={() => onModeChange(m.value)}
            disabled={disabled}
          >
            {m.label}
          </button>
        ))}
        <span className="mode-divider" aria-hidden="true" />
        <button
          type="button"
          role="radio"
          aria-checked={mode === "compare"}
          className={mode === "compare" ? "mode-btn active mode-btn-dev" : "mode-btn mode-btn-dev"}
          onClick={() => onModeChange("compare")}
          disabled={disabled}
        >
          Compare <span className="mode-btn-tag">dev</span>
        </button>
      </div>

      <label className="top-k-label">
        top_k
        <input
          type="number"
          min={1}
          max={50}
          value={topK}
          onChange={handleTopKChange}
          disabled={disabled}
        />
        <span className="field-hint">1–50</span>
      </label>

      <button type="submit" disabled={disabled}>
        Ask
      </button>
    </form>
  );
}
