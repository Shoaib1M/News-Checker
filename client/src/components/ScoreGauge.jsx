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

/*
PURPOSE: Outcomes where a 0-100 "evidence balance" number would be a lie.
WHY THIS EXISTS: The gauge previously drew a number for every status except
`insufficient_evidence`, so a subjective claim or a claim about a future event
got a confident-looking score computed from evidence that does not exist. Each
of these states is a statement about the claim, not a measurement of evidence,
so the dial shows a word instead of a number. Colors stay inside the app's
existing palette — slate for "we can't say", amber for a real negative finding.
*/
const NON_NUMERIC_STATES = {
  insufficient_evidence: { label: "unverified", color: "#64748b" },
  not_a_claim: { label: "no claim", color: "#64748b" },
  not_objectively_verifiable: { label: "subjective", color: "#64748b" },
  not_verifiable_yet: { label: "not yet verifiable", color: "#7c3aed" },
  unsupported_no_coverage: { label: "unsupported", color: "#f59e0b" },
  // The plan is attested; the event has not happened. A number here would be
  // read as "90% true" for something that is not yet true or false at all.
  reported_plan: { label: "reported plan", color: "#7c3aed" },
};

export default function ScoreGauge({ score, assessmentStatus }) {
  const [displayScore, setDisplayScore] = useState(0);
  const [offset, setOffset] = useState(CIRCUMFERENCE); // Start fully empty
  const animationRef = useRef(null);

  useEffect(() => {
    // 1. Animate the colored SVG ring filling up.
    // For a non-numeric outcome the ring stays empty: combined_score is a
    // placeholder 50 in those states, and drawing it as a half-full dial
    // implied a measured "middling" result next to a "—" readout.
    const ringScore = NON_NUMERIC_STATES[assessmentStatus] ? 0 : score;
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

  const nonNumeric = NON_NUMERIC_STATES[assessmentStatus];
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
