/*
FILE PURPOSE:
This component displays three horizontal progress bars that break down the final credibility score.
It shows the user exactly how the ML model, the Web Evidence, and the Stance calculation 
contributed to the final verdict.

FLOW:
1. Receives three scores as props (`mlScore`, `evidenceScore`, `stanceNet`).
2. Converts the raw decimal scores (e.g., 0.85) into percentages (85%).
3. Maps over the items array to render three identical progress bar UI blocks.

WHY THIS EXISTS:
Transparency is crucial in AI. A single number out of 100 isn't enough; the user needs 
to know *why* the AI gave that score.
*/

export default function ScoreBreakdown({ mlScore, evidenceScore, stanceNet }) {
  const items = [
    {
      label: "Model signal",
      value: Math.round(mlScore * 100),
      tooltip: "A legacy learned signal; it is never used alone as proof",
    },
    {
      label: "External evidence",
      value: Math.round(evidenceScore * 100),
      tooltip: "Strength and coverage of retrieved evidence",
    },
    {
      label: "Evidence direction",
      // Stance is usually a number between -1 (Contradicts) and 1 (Supports).
      // We normalize it to a 0-100 scale here so it visually matches the other bars.
      value: Math.round(((stanceNet + 1) / 2) * 100),
      tooltip: "Whether NLI-checked evidence supports or contradicts the claim",
    },
  ];

  return (
    <div className="breakdown-list" id="score-breakdown">
      {items.map((item) => (
        <div className="breakdown-item" key={item.label} title={item.tooltip}>
          <span className="breakdown-label">{item.label}</span>
          <div className="breakdown-bar-track">
            {/* The actual filled portion of the bar */}
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
