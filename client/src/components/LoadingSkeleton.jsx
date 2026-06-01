export default function LoadingSkeleton() {
  return (
    <div className="skeleton-container" id="loading-skeleton">
      {/* Score card skeleton */}
      <div className="score-card" style={{ opacity: 0.7 }}>
        <div className="skeleton-pulse skeleton-gauge" />
        <div style={{ flex: 1 }}>
          <div className="skeleton-pulse skeleton-text" style={{ width: "60%" }} />
          <div className="skeleton-pulse skeleton-text-sm" style={{ width: "80%", marginBottom: "1rem" }} />
          <div className="skeleton-pulse skeleton-bar" style={{ width: "100%", marginBottom: "0.5rem" }} />
          <div className="skeleton-pulse skeleton-bar" style={{ width: "85%", marginBottom: "0.5rem" }} />
          <div className="skeleton-pulse skeleton-bar" style={{ width: "70%" }} />
        </div>
      </div>

      {/* Evidence cards skeleton */}
      <div style={{ marginTop: "1.5rem" }}>
        <div className="skeleton-pulse skeleton-text" style={{ width: "30%", marginBottom: "1rem" }} />
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton-pulse skeleton-card" />
        ))}
      </div>
    </div>
  );
}
