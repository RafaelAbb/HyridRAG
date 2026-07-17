export function QuestionHistory({ entries, onReuse, onClear }) {
  return (
    <details className="history-panel">
      <summary>Recent questions {entries.length > 0 && `(${entries.length})`}</summary>
      {entries.length === 0 && <p className="empty-state">No questions asked yet this session.</p>}
      {entries.length > 0 && (
        <>
          <ul className="history-list">
            {entries.map((e) => (
              <li key={e.id}>
                <button type="button" className="history-item" onClick={() => onReuse(e)}>
                  <span className="history-question">{e.question}</span>
                  <span className="history-meta">
                    {e.mode} · top_k={e.topK}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <button type="button" className="history-clear" onClick={onClear}>
            Clear history
          </button>
        </>
      )}
    </details>
  );
}
