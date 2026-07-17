import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion } from "../api/client.js";

// One instance per retrieval mode — kept independent so hybrid/dense/sparse
// panels load/error independently. Dense-only usually resolves first since
// it skips the cross-encoder rerank step.
export function useAsk(mode) {
  const [state, setState] = useState({ status: "idle", data: null, error: null, tookMs: null });
  const controllerRef = useRef(null);

  const ask = useCallback(
    async (question, topK) => {
      // A new ask() supersedes whatever this hook instance was already
      // waiting on — abort it so a slow, now-irrelevant response can't
      // overwrite the newer one when it eventually resolves.
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;

      setState({ status: "loading", data: null, error: null, tookMs: null });
      const start = performance.now();
      try {
        const data = await askQuestion({ question, topK, retrievalMode: mode, signal: controller.signal });
        setState({ status: "success", data, error: null, tookMs: Math.round(performance.now() - start) });
      } catch (err) {
        if (err.name === "AbortError") return; // superseded — leave no trace, newer request owns state now
        setState({ status: "error", data: null, error: err.message, tookMs: null });
      }
    },
    [mode]
  );

  useEffect(() => () => controllerRef.current?.abort(), []);

  return { ...state, ask };
}
