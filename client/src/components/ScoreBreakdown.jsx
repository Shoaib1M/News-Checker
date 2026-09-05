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

export default function ScoreBreakdown({ mlScore, evidenceScore, stanceNet, hasClassifiedEvidence }) {
  const evidencePct = Math.round(evidenceScore * 100);
  const directionPct = Math.round(((stanceNet + 1) / 2) * 100);

  const items = [
    {
      label: "Model signal",
      value: Math.round(mlScore * 100),
      display: `${Math.round(mlScore * 100)}%`,
      tooltip: "A legacy learned signal; it is never used alone as proof",
    },
    {
      label: "External evidence",
      // Without any NLI-classified evidence, a percentage here would be fake
      // precision — there is nothing to measure yet, not a low score.
      value: hasClassifiedEvidence ? evidencePct : 0,
      display: hasClassifiedEvidence ? `${evidencePct}%` : "—",
      tooltip: "Strength and coverage of NLI-classified evidence",
    },
    {
      label: "Evidence direction",
      value: hasClassifiedEvidence ? directionPct : 0,
      display: hasClassifiedEvidence ? `${directionPct}%` : "Not available",
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
          <span className="breakdown-value">{item.display}</span>
        </div>
      ))}
    </div>
  );
}
