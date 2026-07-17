import { useCallback, useState } from "react";

const STORAGE_KEY = "rag-dashboard:history";
const MAX_ENTRIES = 20;

function loadEntries() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    // Private browsing / storage disabled / corrupted JSON — degrade to
    // empty history rather than crashing the Ask flow over this.
    return [];
  }
}

function persist(entries) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Storage full/blocked — history still works in-memory for this session.
  }
}

// HistoryEntry: { id, question, topK, mode, askedAt }
// Deliberately not storing the answer itself — history is for re-asking,
// not for browsing past responses.
export function useQuestionHistory() {
  const [entries, setEntries] = useState(loadEntries);

  const addEntry = useCallback((question, topK, mode) => {
    setEntries((prev) => {
      const next = [
        { id: crypto.randomUUID(), question, topK, mode, askedAt: Date.now() },
        ...prev,
      ].slice(0, MAX_ENTRIES);
      persist(next);
      return next;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setEntries([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }, []);

  return { entries, addEntry, clearHistory };
}
