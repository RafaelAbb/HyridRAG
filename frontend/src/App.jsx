import { useEffect, useState } from "react";
import { fetchHealth } from "./api/client.js";
import { QuestionForm } from "./components/QuestionForm.jsx";
import { QuestionHistory } from "./components/QuestionHistory.jsx";
import { ComparisonView } from "./components/ComparisonView.jsx";
import { SingleAnswerView } from "./components/SingleAnswerView.jsx";
import { IngestForm } from "./components/IngestForm.jsx";
import { DocumentsPanel } from "./components/DocumentsPanel.jsx";
import { useQuestionHistory } from "./hooks/useQuestionHistory.js";

const DEFAULT_FORM = { question: "", topK: 5, mode: "hybrid" };

export default function App() {
  const [tab, setTab] = useState("ask");
  const [backendReady, setBackendReady] = useState(true);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [asked, setAsked] = useState(null); // { question, topK, mode }
  const [refreshKey, setRefreshKey] = useState(0);
  const { entries, addEntry, clearHistory } = useQuestionHistory();

  useEffect(() => {
    let cancelled = false;
    let timer;

    function check() {
      fetchHealth()
        .then(() => {
          if (!cancelled) setBackendReady(true);
        })
        .catch(() => {
          if (cancelled) return;
          setBackendReady(false);
          timer = setTimeout(check, 2000);
        });
    }

    check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  function handleAsk(question, topK, mode) {
    setAsked({ question, topK, mode });
    addEntry(question, topK, mode);
  }

  function handleReuse(entry) {
    // Re-populate the form AND re-fire immediately — a history entry is by
    // definition a previously-valid submission, so there's no reason to
    // make the user confirm with a second click.
    setForm({ question: entry.question, topK: entry.topK, mode: entry.mode });
    setAsked({ question: entry.question, topK: entry.topK, mode: entry.mode });
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <p className="app-eyebrow">Portfolio · RAG System</p>
          <h1>RAG Dashboard</h1>
        </div>
        <nav className="tabs">
          <button className={tab === "ask" ? "tab active" : "tab"} onClick={() => setTab("ask")}>
            Ask
          </button>
          <button className={tab === "manage" ? "tab active" : "tab"} onClick={() => setTab("manage")}>
            Manage Index
          </button>
        </nav>
      </header>

      {!backendReady && <div className="banner">Backend starting up — retrying...</div>}

      {/*
        display:none, not conditional unmount — unmounting on tab switch
        destroys each useAsk instance's state, so returning to the Ask tab
        would silently re-fire the same request for an answer already had.
        Both sections stay mounted; the trade-off is both mount eagerly on
        load instead of lazily, which is cheap here.
      */}
      <section style={{ display: tab === "ask" ? undefined : "none" }}>
        <QuestionForm
          question={form.question}
          topK={form.topK}
          mode={form.mode}
          onQuestionChange={(question) => setForm((f) => ({ ...f, question }))}
          onTopKChange={(topK) => setForm((f) => ({ ...f, topK }))}
          onModeChange={(mode) => setForm((f) => ({ ...f, mode }))}
          onSubmit={handleAsk}
          disabled={!backendReady}
        />
        <QuestionHistory entries={entries} onReuse={handleReuse} onClear={clearHistory} />

        {!asked && (
          <p className="empty-state">Ask a question above to see a grounded answer with citations.</p>
        )}
        {asked && asked.mode === "compare" && (
          <ComparisonView question={asked.question} topK={asked.topK} />
        )}
        {asked && asked.mode !== "compare" && (
          <SingleAnswerView question={asked.question} topK={asked.topK} mode={asked.mode} />
        )}
      </section>

      <section className="manage-section" style={{ display: tab === "manage" ? undefined : "none" }}>
        <IngestForm onIngested={() => setRefreshKey((k) => k + 1)} />
        <DocumentsPanel refreshKey={refreshKey} />
      </section>
    </div>
  );
}
