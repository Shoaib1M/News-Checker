export default function ScoreBreakdown({ mlScore, evidenceScore, stanceNet }) {
  const items = [
    {
      label: "ML Model",
      value: Math.round(mlScore * 100),
      tooltip: "Confidence from the trained neural network",
    },
    {
      label: "Evidence",
      value: Math.round(evidenceScore * 100),
      tooltip: "Similarity to scraped web evidence",
    },
    {
      label: "Stance",
      value: Math.round(((stanceNet + 1) / 2) * 100),
      tooltip: "Whether evidence supports or contradicts the claim",
    },
  ];

  return (
    <div className="breakdown-list" id="score-breakdown">
      {items.map((item) => (
        <div className="breakdown-item" key={item.label} title={item.tooltip}>
          <span className="breakdown-label">{item.label}</span>
          <div className="breakdown-bar-track">
            <div
              className="breakdown-bar-fill"
              style={{ width: `${item.value}%` }}
            />
          </div>
          <span className="breakdown-value">{item.value}%</span>
        </div>
      ))}
    </div>
  );
}
