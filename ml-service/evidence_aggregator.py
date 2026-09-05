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
from typing import Iterable
from urllib.parse import urlparse


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
    publisher: str = ""


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


def count_independent_groups(results: Iterable) -> int:
    """Count distinct independent source origins among NLI-classified evidence.

    Several articles from the same publisher domain — or syndicated copies —
    count as one independent group, not several.  Near-duplicate titles are
    already deduplicated upstream in ``providers/registry.py``; this counts
    the remaining distinct publisher domains so the same outlet's coverage
    can't be presented as multiple independent confirmations.

    The publisher host is preferred over the URL host: an article reached
    through an aggregator carries the aggregator's hostname, so counting by
    URL would file ten different newsrooms as a single origin.

    Accepts anything with ``.url`` and ``.nli_available`` attributes
    (``ClassifiedEvidence`` or ``evidence_pipeline.EvidenceResult``).
    """
    domains: set[str] = set()
    for r in results:
        if not getattr(r, "nli_available", False):
            continue
        host = getattr(r, "publisher", "") or urlparse(r.url).netloc.lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.add(host)
    return len(domains)


# ── Absence-of-evidence reasoning ────────────────────────────────────
# A search that ran correctly and turned up nothing supporting the claim is
# not the same event as a search that failed. For a claim whose true version
# would necessarily have been reported everywhere, the first case is a real
# finding; the second is still just "we don't know". Conflating them was the
# single biggest correctness gap in this pipeline: an obviously fabricated
# headline and a broken DuckDuckGo request produced the identical verdict.
#
# This is deliberately narrow. All four conditions must hold, because the
# cost of a false positive here is asserting that nobody reported something
# when in fact we simply failed to look properly.
MIN_CANDIDATES_FOR_ABSENCE = 4

# NO_RELEVANT_RESULTS belongs here: it means the search ran and returned a
# pool of candidates, and the relevance filter found that none of them
# discussed what the claim asserts. That is the *canonical* shape of an
# absence-of-coverage finding, not a retrieval failure. SEARCH_FAILED and
# NO_RESULTS stay out — those tell us nothing about the world.
_SEARCH_WORKED = {"SEARCH_SUCCESS", "SEARCH_PARTIAL", "NO_RELEVANT_RESULTS"}


def assess_coverage(
    stance: dict,
    retrieval_status: str,
    candidate_count: int,
    salience: str,
    prospective: bool = False,
    nli_ready: bool = True,
    negated: bool = False,
) -> dict | None:
    """Return an overriding stance when *absence of coverage* is itself evidence.

    Returns ``None`` — leaving the caller's existing stance untouched — unless
    every one of these holds:

    1. The search actually ran (``SEARCH_SUCCESS`` / ``SEARCH_PARTIAL``).
       A failed or timed-out search tells us nothing about the world.
    2. It returned a real pool of candidates (``MIN_CANDIDATES_FOR_ABSENCE``).
       Two results is a thin search, not a canvass of the press.
    3. Nothing in that pool supported the claim, and nothing contradicted it
       either — a contradiction is stronger evidence and should win on its own.
    4. The claim is high-salience: a true version of it could not have gone
       unreported (see claim_triage).
    5. The claim is not negated. Absence of coverage cannot count against a
       claim that something did *not* happen — no outlet reporting that the
       US banned Google is exactly what "the US did not ban Google" predicts,
       so treating silence as evidence against it inverts the inference.
    6. The NLI model was available. "No retrieved source supports this" is a
       claim about what the sources say, and we only know what they say if
       something actually read them. With NLI down we compared nothing, so
       the honest answer stays "could not verify".

    Parameters
    ----------
    prospective:
        True when the claim describes a future event. The wording changes —
        the finding is that no source reports the *plan*, not that no source
        reports the event — but the logic is identical.
    """
    if negated:
        return None
    if not nli_ready:
        return None
    if retrieval_status not in _SEARCH_WORKED:
        return None
    if candidate_count < MIN_CANDIDATES_FOR_ABSENCE:
        return None
    if salience != "high":
        return None
    if stance.get("supporting_count", 0) > 0 or stance.get("contradicting_count", 0) > 0:
        return None

    subject = "such a plan" if prospective else "this"
    return {
        **stance,
        "status": "unsupported_no_coverage",
        "verdict": f"no credible source reports {subject}",
        "coverage_checked": candidate_count,
    }
