import { useEffect } from "react";
import { useAsk } from "../hooks/useAsk.js";
import { AnswerPanel } from "./AnswerPanel.jsx";

export const MODE_LABELS = {
  hybrid: "Hybrid (dense + sparse + rerank)",
  dense: "Dense-only",
  sparse: "Sparse-only",
};

export function SingleAnswerView({ question, topK, mode }) {
  const ask = useAsk(mode);

  useEffect(() => {
    if (!question) return;
    ask.ask(question, topK);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [question, topK, mode]);

  return <AnswerPanel title={MODE_LABELS[mode]} {...ask} />;
}
