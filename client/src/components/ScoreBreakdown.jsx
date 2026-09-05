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

export default function ScoreBreakdown({
  mlScore,
  evidenceScore,
  stanceNet,
  hasClassifiedEvidence,
  hasDirectionalEvidence,
}) {
  const evidencePct = Math.round(evidenceScore * 100);
  const directionPct = Math.round(((stanceNet + 1) / 2) * 100);

  const items = [
    {
      // Listed last and labelled for what it is. This MLP was trained on the
      // LIAR corpus of US political statements; on anything else its output
      // is a number without meaning. It contributes nothing to the verdict —
      // the backend computes the verdict from evidence alone — and showing it
      // first invited the reading that the app is "an ML model that scores
      // claims", which is exactly the wrong mental model.
      label: "Legacy ML prior",
      value: Math.round(mlScore * 100),
      display: `${Math.round(mlScore * 100)}%`,
      tooltip:
        "A classifier trained on the LIAR political-statement dataset. Shown for transparency only — it never affects the verdict.",
      muted: true,
    },
    {
      label: "Evidence strength",
      // Without any NLI-classified evidence, a percentage here would be fake
      // precision — there is nothing to measure yet, not a low score.
      value: hasClassifiedEvidence ? evidencePct : 0,
      display: hasClassifiedEvidence ? `${evidencePct}%` : "—",
      tooltip: "Strength and coverage of NLI-classified evidence",
    },
    {
      // Gated on *directional* evidence, not merely classified evidence.
      // Sources the model classified as neutral leave stanceNet at 0, which
      // rendered as a confident-looking half-filled bar at "50%" — a made-up
      // midpoint for a claim nothing had taken a position on.
      label: "Evidence direction",
      value: hasDirectionalEvidence ? directionPct : 0,
      display: hasDirectionalEvidence ? `${directionPct}%` : "—",
      tooltip: "Whether NLI-checked evidence supports or contradicts the claim",
    },
  ];

  // Evidence-derived rows first; the legacy prior sinks to the bottom.
  const ordered = [...items].sort((a, b) => Number(!!a.muted) - Number(!!b.muted));

  return (
    <div className="breakdown-list" id="score-breakdown">
      {ordered.map((item) => (
        <div
          className={`breakdown-item${item.muted ? " breakdown-item-muted" : ""}`}
          key={item.label}
          title={item.tooltip}
        >
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
      <p className="breakdown-caption">
        The verdict is computed from evidence only. The legacy ML prior is displayed
        for transparency and is not part of it.
      </p>
    </div>
  );
}
