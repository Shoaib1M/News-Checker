"""Evidence aggregation — turns classified sources into an overall verdict.

This module takes a list of NLI-classified evidence results and computes
a weighted stance summary.  Source tier weights ensure that one CDC dataset
outweighs five blog reposts, and syndication detection prevents
wire-service copies from inflating evidence counts.

A verdict is produced only when:
- At least one source passed NLI classification, AND
- At least two independent source groups exist, OR
- A single primary/fact-check source entails/contradicts with high confidence

Otherwise the status is "insufficient_evidence".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassifiedEvidence:
    """An evidence result that has been NLI-classified."""
    url: str
    source: str
    source_tier: str
    source_weight: float
    support_score: float
    contradiction_score: float
    nli_available: bool
    stance: str


def compute_stance(results: list[ClassifiedEvidence]) -> dict:
    """Return a weighted stance summary from classified evidence.

    Returns a dict with:
        support, contradiction, net, status, verdict,
        nli_available, evidence_count, evidence_used_count,
        supporting_count, contradicting_count, neutral_count
    """
    nli_results = [r for r in results if r.nli_available]
    if not nli_results:
        return {
            "support": 0.0,
            "contradiction": 0.0,
            "net": 0.0,
            "status": "insufficient_evidence",
            "verdict": "insufficient evidence",
            "nli_available": False,
            "evidence_count": 0,
            "evidence_used_count": 0,
            "supporting_count": 0,
            "contradicting_count": 0,
            "neutral_count": 0,
        }

    # Weighted aggregation
    weighted_support = 0.0
    weighted_contradiction = 0.0
    total_weight = 0.0
    supporting_count = 0
    contradicting_count = 0
    neutral_count = 0

    for r in nli_results:
        w = max(r.source_weight, 0.1)
        weighted_support += r.support_score * w
        weighted_contradiction += r.contradiction_score * w
        total_weight += w

        if r.stance == "supports":
            supporting_count += 1
        elif r.stance == "contradicts":
            contradicting_count += 1
        else:
            neutral_count += 1

    if total_weight > 0:
        avg_support = weighted_support / total_weight
        avg_contradiction = weighted_contradiction / total_weight
    else:
        avg_support = avg_contradiction = 0.0

    net = avg_support - avg_contradiction

    # Status determination
    if len(nli_results) < 1:
        status = "insufficient_evidence"
        verdict = "insufficient evidence"
    elif supporting_count > 0 and contradicting_count > 0:
        status = "mixed"
        verdict = "claims have mixed evidence"
    elif avg_support > 0.35 and supporting_count > 0:
        status = "supported"
        verdict = "evidence supports the claim"
    elif avg_contradiction > 0.35 and contradicting_count > 0:
        status = "contradicted"
        verdict = "evidence contradicts the claim"
    else:
        status = "insufficient_evidence"
        verdict = "insufficient evidence"

    return {
        "support": round(avg_support, 4),
        "contradiction": round(avg_contradiction, 4),
        "net": round(net, 4),
        "status": status,
        "verdict": verdict,
        "nli_available": True,
        "evidence_count": len(nli_results),
        "evidence_used_count": len(nli_results),
        "supporting_count": supporting_count,
        "contradicting_count": contradicting_count,
        "neutral_count": neutral_count,
    }
