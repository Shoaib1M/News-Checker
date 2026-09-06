/*
FILE PURPOSE:
The single list of verdict outcomes that must NOT be drawn as a number, with
the word and colour each one shows instead.

WHY THIS EXISTS:
`combined_score` is an evidence-balance dial. For several outcomes there is no
evidence balance to report — the result is a statement about the claim, not a
measurement of evidence for it — and those outcomes carry a placeholder 50
that renders as a confident, amber "middling" score.

This list lived inside ScoreGauge, and HistoryPanel had its own version of the
same idea that only knew about `insufficient_evidence`. So a saved check of
"asdkjh asdkjh" appeared in the history list as **50** in amber, as though it
were a half-true claim, and so did "no credible source reports this".

That is the third time in this codebase the same rule has been duplicated and
the copies have drifted (the others were the subjective-superlative pattern,
which existed in both claim_triage and knowledge_verifier, and the independent
publisher count). Hence one module, imported by both.

Mirrors NON_NUMERIC_STATUSES in ml-service/main.py, which is where the backend
enforces the same rule; test_claim_edge_cases.py pins that side.
*/

// status -> { label, color }. Colours stay inside the existing palette:
// slate for "we can't say", amber for a real negative finding, purple for
// "not yet decidable".
export const NON_NUMERIC_STATES = {
  insufficient_evidence: { label: "unverified", color: "#64748b" },
  not_a_claim: { label: "no claim", color: "#64748b" },
  not_objectively_verifiable: { label: "subjective", color: "#64748b" },
  not_verifiable_yet: { label: "not yet verifiable", color: "#7c3aed" },
  unsupported_no_coverage: { label: "unsupported", color: "#f59e0b" },
  // Real evidence exists, but it attests the announcement rather than the
  // event. A number here would read as "90% true" for something that is not
  // yet true or false at all.
  reported_plan: { label: "reported plan", color: "#7c3aed" },
};

/*
PURPOSE: Whether this outcome may be shown as a 0-100 number at all.
*/
export function isNumericVerdict(status) {
  return !NON_NUMERIC_STATES[status];
}

/*
PURPOSE: The word to show in place of a number, or null if a number is fine.
*/
export function verdictStateFor(status) {
  return NON_NUMERIC_STATES[status] || null;
}
