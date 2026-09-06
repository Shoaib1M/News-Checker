/*
FILE PURPOSE:
A visually appealing, animated circular gauge that displays the final 0-100 score.

FLOW:
1. Calculates the SVG circumference to animate the stroke (the colored ring).
2. Uses `requestAnimationFrame` to smoothly count the number up from 0 to the final score.
3. Sets the color of the ring dynamically based on how high the score is (Red -> Green).

WHY THIS EXISTS:
A large, animated dial provides immediate visual feedback. The animation makes the app feel
polished, dynamic, and modern, which builds user trust.
*/

import { useEffect, useRef, useState } from "react";

// Shared with HistoryPanel: the same outcome must never be a word in one
// place and a number in another.
import { verdictStateFor } from "../verdictStates";

const CIRCUMFERENCE = 2 * Math.PI * 90; // Radius of the circle is 90

/*
PURPOSE: Returns a hex color code based on the score threshold.
WHY THIS EXISTS: So the user can immediately tell if a score is "good" or "bad" without reading the text.
*/
function getStrokeColor(score) {
  if (score <= 25) return "#ef4444"; // Red
  if (score <= 40) return "#f97316"; // Orange
  if (score <= 60) return "#f59e0b"; // Yellow
  if (score <= 75) return "#10b981"; // Light Green
  return "#059669"; // Dark Green
}

export default function ScoreGauge({ score, assessmentStatus }) {
  const [displayScore, setDisplayScore] = useState(0);
  const [offset, setOffset] = useState(CIRCUMFERENCE); // Start fully empty
  const animationRef = useRef(null);

  useEffect(() => {
    // 1. Animate the colored SVG ring filling up.
    // For a non-numeric outcome the ring stays empty: combined_score is a
    // placeholder 50 in those states, and drawing it as a half-full dial
    // implied a measured "middling" result next to a "—" readout.
    const ringScore = verdictStateFor(assessmentStatus) ? 0 : score;
    const target = CIRCUMFERENCE - (ringScore / 100) * CIRCUMFERENCE;
    // Small delay ensures the animation doesn't finish before the browser actually renders the element
    const timer = setTimeout(() => setOffset(target), 100);

    // 2. Animate the text number counting up from 0
    const duration = 1500; // 1.5 seconds
    const start = performance.now();

    const animate = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      
      // Math trick (cubic ease-out) to make the counter slow down as it gets closer to the final number
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayScore(Math.round(eased * score));

      // Keep animating until progress hits 100%
      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };
    
    // Start the animation loop
    animationRef.current = requestAnimationFrame(animate);

    // Cleanup function: Stops animations if the component is destroyed before they finish
    return () => {
      clearTimeout(timer);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [score, assessmentStatus]);

  const nonNumeric = verdictStateFor(assessmentStatus);
  const color = nonNumeric ? nonNumeric.color : getStrokeColor(score);

  return (
    <div className="gauge-container" id="score-gauge">
      <svg className="gauge-svg" viewBox="0 0 200 200">
        {/* The faint background ring */}
        <circle className="gauge-bg" cx="100" cy="100" r="90" />
        
        {/* The brightly colored progress ring */}
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
          {nonNumeric ? "—" : displayScore}
        </div>
        <div className="gauge-label">
          {nonNumeric ? nonNumeric.label : "evidence balance"}
        </div>
      </div>
    </div>
  );
}
