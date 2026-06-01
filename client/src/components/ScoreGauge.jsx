import { useEffect, useRef, useState } from "react";

const CIRCUMFERENCE = 2 * Math.PI * 90; // radius = 90

function getStrokeColor(score) {
  if (score <= 25) return "#ef4444";
  if (score <= 40) return "#f97316";
  if (score <= 60) return "#f59e0b";
  if (score <= 75) return "#10b981";
  return "#059669";
}

export default function ScoreGauge({ score, verdict }) {
  const [displayScore, setDisplayScore] = useState(0);
  const [offset, setOffset] = useState(CIRCUMFERENCE);
  const animationRef = useRef(null);

  useEffect(() => {
    // Animate the gauge fill
    const target = CIRCUMFERENCE - (score / 100) * CIRCUMFERENCE;
    // Small delay so the animation is visible after mount
    const timer = setTimeout(() => setOffset(target), 100);

    // Animate the number count-up
    const duration = 1500;
    const start = performance.now();
    const animate = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayScore(Math.round(eased * score));
      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      clearTimeout(timer);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [score]);

  const color = getStrokeColor(score);

  return (
    <div className="gauge-container" id="score-gauge">
      <svg className="gauge-svg" viewBox="0 0 200 200">
        <circle className="gauge-bg" cx="100" cy="100" r="90" />
        <circle
          className="gauge-fill"
          cx="100"
          cy="100"
          r="90"
          stroke={color}
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={offset}
          style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
        />
      </svg>
      <div className="gauge-score-text">
        <div className="gauge-number" style={{ color }}>
          {displayScore}
        </div>
        <div className="gauge-label">out of 100</div>
      </div>
    </div>
  );
}
