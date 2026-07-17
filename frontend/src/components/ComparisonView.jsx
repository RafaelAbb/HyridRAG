import { useEffect } from "react";
import { useAsk } from "../hooks/useAsk.js";
import { AnswerPanel } from "./AnswerPanel.jsx";
import { MODE_LABELS } from "./SingleAnswerView.jsx";

// Development comparison — runs all three retrieval strategies against the
// same question so the retrieval trade-offs (hybrid's rerank quality vs
// dense/sparse's lower latency) are visible directly, side by side.
//
// Three named useAsk() calls, not a modes[] prop: React hooks can't be
// called a variable number of times per render, and the comparison is a
// fixed 3-way (hybrid/dense/sparse) by design, not an open-ended N.
export function ComparisonView({ question, topK }) {
  const hybrid = useAsk("hybrid");
  const dense = useAsk("dense");
  const sparse = useAsk("sparse");

  useEffect(() => {
    if (!question) return;
    hybrid.ask(question, topK);
    dense.ask(question, topK);
    sparse.ask(question, topK);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [question, topK]);

  return (
    <div>
      <p className="compare-hint">
        Development comparison — runs all three retrieval strategies against the same question
        so you can see the retrieval trade-offs directly.
      </p>
      <div className="comparison-grid">
        <AnswerPanel title={MODE_LABELS.hybrid} {...hybrid} />
        <AnswerPanel title={MODE_LABELS.dense} {...dense} />
        <AnswerPanel title={MODE_LABELS.sparse} {...sparse} />
      </div>
    </div>
  );
}
