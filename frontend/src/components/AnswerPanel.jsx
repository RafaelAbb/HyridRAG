import { useState } from "react";
import { ConfidenceMeter } from "./ConfidenceMeter.jsx";
import { ChunkList } from "./ChunkList.jsx";

// Splits "...claim [2] more text [1]..." into plain-text segments and
// citation markers. Citation numbers are 1-based and map to sources[n-1]
// (see the contract note in api/client.js).
function renderAnswerWithCitations(answer, onCiteClick) {
  const parts = answer.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) return <span key={i}>{part}</span>;
    const n = Number(match[1]);
    return (
      <sup key={i}>
        {/*
          A real <button>, not role="button"+tabIndex — role="button" is
          focusable and announced correctly, but does NOT grant Enter/Space
          activation for free. A native button does, with zero extra JS.
        */}
        <button type="button" className="citation-chip" onClick={() => onCiteClick(n - 1)}>
          [{n}]
        </button>
      </sup>
    );
  });
}

function AnswerSkeleton() {
  return (
    <div className="answer-skeleton" aria-hidden="true">
      <div className="skeleton-line skeleton-line-full" />
      <div className="skeleton-line skeleton-line-full" />
      <div className="skeleton-line skeleton-line-short" />
      <div className="skeleton-bar" />
      <div className="skeleton-line skeleton-line-medium" />
      <div className="skeleton-line skeleton-line-medium" />
    </div>
  );
}

function statusAnnouncement(status, data, error) {
  if (status === "loading") return "Loading answer";
  if (status === "success" && data?.has_answer) return "Answer ready";
  if (status === "success" && data && !data.has_answer) return "No answer found";
  if (status === "error") return `Error: ${error}`;
  return "";
}

export function AnswerPanel({ title, status, data, error, tookMs }) {
  const [highlightedIndex, setHighlightedIndex] = useState(null);

  return (
    <div className="answer-panel">
      <div className="answer-panel-header">
        <h3 className="answer-panel-title">{title}</h3>
        {status === "success" && tookMs != null && <span className="latency-badge">{tookMs}ms</span>}
      </div>

      {/* Announces status changes for screen readers; the skeleton/visual states below are aria-hidden or self-descriptive. */}
      <div aria-live="polite" className="sr-only">
        {statusAnnouncement(status, data, error)}
      </div>

      {status === "idle" && <p className="muted">Ask a question to see this panel populate.</p>}
      {status === "loading" && <AnswerSkeleton />}
      {status === "error" && (
        <p className="error-alert" role="alert">
          Error: {error}
        </p>
      )}

      {status === "success" && data && !data.has_answer && (
        <p className="no-answer">No answer found in the indexed documents.</p>
      )}

      {status === "success" && data && data.has_answer && (
        <>
          <p className="answer-text">
            {renderAnswerWithCitations(data.answer, setHighlightedIndex)}
          </p>
          <ConfidenceMeter confidence={data.confidence} />
          <ChunkList sources={data.sources} highlightedIndex={highlightedIndex} />
        </>
      )}
    </div>
  );
}
