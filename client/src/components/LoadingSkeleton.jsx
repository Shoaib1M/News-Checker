/*
FILE PURPOSE:
Displays a "skeleton" loading state while waiting for the backend to finish processing a fact-check.

FLOW:
1. Renders empty gray boxes that mimic the shape of the final result UI.
2. CSS animations (`.skeleton-pulse` in index.css) make these boxes pulse, indicating activity.

WHY THIS EXISTS:
A skeleton loader provides better "perceived performance" than a simple spinning wheel. 
It prepares the user's eyes for where the data will appear, making the wait feel shorter.
*/

export default function LoadingSkeleton() {
  return (
    <div className="skeleton-container" id="loading-skeleton">
      {/* Score card skeleton (Top Section) */}
      <div className="score-card" style={{ opacity: 0.7 }}>
        {/* Simulates the circular ScoreGauge */}
        <div className="skeleton-pulse skeleton-gauge" />
        <div style={{ flex: 1 }}>
          {/* Simulates text lines */}
          <div className="skeleton-pulse skeleton-text" style={{ width: "60%" }} />
          <div className="skeleton-pulse skeleton-text-sm" style={{ width: "80%", marginBottom: "1rem" }} />
          {/* Simulates the ScoreBreakdown bars */}
          <div className="skeleton-pulse skeleton-bar" style={{ width: "100%", marginBottom: "0.5rem" }} />
          <div className="skeleton-pulse skeleton-bar" style={{ width: "85%", marginBottom: "0.5rem" }} />
          <div className="skeleton-pulse skeleton-bar" style={{ width: "70%" }} />
        </div>
      </div>

      {/* Evidence cards skeleton (Bottom Section) */}
      <div style={{ marginTop: "1.5rem" }}>
        <div className="skeleton-pulse skeleton-text" style={{ width: "30%", marginBottom: "1rem" }} />
        {/* Render 3 fake evidence cards */}
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton-pulse skeleton-card" />
        ))}
      </div>
    </div>
  );
}
