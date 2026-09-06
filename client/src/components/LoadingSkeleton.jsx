/*
FILE PURPOSE:
Displays a "skeleton" loading state while waiting for the backend to finish processing a fact-check.

FLOW:
1. Renders empty gray boxes that mimic the shape of the final result UI.
2. CSS animations (`.skeleton-pulse` in index.css) make these boxes pulse, indicating activity.

WHY THIS EXISTS:
A skeleton loader provides better "perceived performance" than a simple spinning wheel. 
It prepares the user's eyes for where the data will appear, making the wait feel shorter.

It also reports elapsed time and, past a threshold, explains WHY the wait is
long. A check involves several live searches and, on the first request of a
process's life, downloading and loading the NLI model — which can take a
minute or more. Without that explanation the pause is indistinguishable from
the app having hung, which is the impression a first-time viewer forms.
*/

import { useEffect, useState } from "react";

// Past this many seconds, silence starts to read as a hang.
const EXPLAIN_AFTER_SECONDS = 8;

export default function LoadingSkeleton() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)),
      1000,
    );
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="skeleton-container" id="loading-skeleton">
      <div className="skeleton-status">
        <span className="skeleton-elapsed">{elapsed}s</span>
        <span>
          {elapsed < EXPLAIN_AFTER_SECONDS
            ? "Searching for evidence…"
            : "Still working — this runs several live searches, and the first "
              + "check of a session also loads the NLI model, which can take a minute."}
        </span>
      </div>

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
