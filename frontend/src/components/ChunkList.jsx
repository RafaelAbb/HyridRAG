export function ChunkList({ sources, highlightedIndex }) {
  if (!sources.length) return null;

  return (
    <ol className="chunk-list">
      {sources.map((source, i) => (
        <li
          key={source.doc_id}
          className={i === highlightedIndex ? "chunk-item chunk-highlighted" : "chunk-item"}
        >
          <span className="chunk-source-name">{source.source_name}</span>
          <span className="chunk-score">score={source.score.toFixed(3)}</span>
        </li>
      ))}
    </ol>
  );
}
