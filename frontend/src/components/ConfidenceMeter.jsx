export function ConfidenceMeter({ confidence }) {
  const pct = Math.round(confidence * 100);
  const band = confidence >= 0.66 ? "high" : confidence >= 0.33 ? "medium" : "low";

  return (
    <div className="confidence-meter">
      <div className="confidence-track">
        <div className={`confidence-fill confidence-${band}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="confidence-label">{pct}% confidence</span>
    </div>
  );
}
