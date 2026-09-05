"""Evidence aggregation — turns classified sources into an overall verdict.

This module takes a list of NLI-classified evidence results and computes
a weighted stance summary.  Source tier weights ensure that one CDC dataset
outweighs five blog reposts, and syndication detection prevents
wire-service copies from inflating evidence counts.

A directional verdict requires at least one source that NLI classified as
entailing or contradicting the claim, with a weighted mean score above
MIN_DIRECTIONAL_STRENGTH. Otherwise the status is "insufficient_evidence".

Independence is not a gate on the verdict — one Reuters article that clearly
entails a claim is real evidence — but it *is* what drives confidence. The
distinct-publisher counts returned here are what stop four copies of one wire
story from reading as four confirmations.
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


def _publisher_of(result: ClassifiedEvidence) -> str:
    """Identify who published this, preferring the resolved publisher host."""
    host = result.publisher or urlparse(result.url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


# A direction only wins outright when its weighted mass is at least this many
# times the opposing side's. Below that the evidence is genuinely contested
# and the honest answer is "mixed".
DOMINANCE_RATIO = 2.0

# Minimum weighted mean score for a direction to count at all.
MIN_DIRECTIONAL_STRENGTH = 0.35


def compute_stance(results: list[ClassifiedEvidence]) -> dict:
    """Return a weighted stance summary from classified evidence.

    Returns a dict with:
        support, contradiction, net, status, verdict,
        nli_available, evidence_count, evidence_used_count,
        supporting_count, contradicting_count, neutral_count,
        independent_supporting, independent_contradicting

    HOW THE DIRECTION IS SCORED — and why it changed:
    ``support`` and ``contradiction`` used to be weighted means over *every*
    classified source, neutrals included. That made the verdict
    non-monotonic in a way nobody would expect: one Reuters article entailing
    the claim at 0.93 gave ``supported``, and adding three on-topic articles
    that said nothing either way dragged the mean to 0.26 and turned the same
    evidence into ``insufficient_evidence``. More evidence, none of it
    disagreeing, made the system less certain.

    Each direction is now scored over the sources that actually take it.
    Neutral coverage still counts — as the evidence pool the aggregator
    canvassed, and in the confidence calculation — but it no longer dilutes
    a conclusion it does not contradict.
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
            "independent_supporting": 0,
            "independent_contradicting": 0,
        }

    supporting = [r for r in nli_results if r.stance == "supports"]
    contradicting = [r for r in nli_results if r.stance == "contradicts"]
    neutral = [r for r in nli_results if r.stance not in {"supports", "contradicts"}]

    def weighted(side: list[ClassifiedEvidence], attribute: str) -> tuple[float, float]:
        """Return (weighted mean score, total weighted mass) for one side."""
        if not side:
            return 0.0, 0.0
        total_weight = sum(max(r.source_weight, 0.1) for r in side)
        mass = sum(getattr(r, attribute) * max(r.source_weight, 0.1) for r in side)
        return (mass / total_weight if total_weight else 0.0), mass

    avg_support, support_mass = weighted(supporting, "support_score")
    avg_contradiction, contradiction_mass = weighted(contradicting, "contradiction_score")
    net = avg_support - avg_contradiction

    # Distinct publishers per direction. Four copies of one wire story from
    # one newsroom are one confirmation, not four — the syndication guard the
    # module has always claimed and did not previously apply.
    independent_supporting = len({_publisher_of(r) for r in supporting})
    independent_contradicting = len({_publisher_of(r) for r in contradicting})

    support_ok = supporting and avg_support > MIN_DIRECTIONAL_STRENGTH
    contradiction_ok = contradicting and avg_contradiction > MIN_DIRECTIONAL_STRENGTH

    if support_ok and contradiction_ok:
        # Both directions are represented. Whether that is genuinely contested
        # or one weak dissent against a solid consensus depends on the
        # weighted mass, not on the raw counts: five strong reports from
        # reputable outlets used to be filed as "mixed" against one 0.40
        # contradiction from an unclassified blog.
        if support_mass >= contradiction_mass * DOMINANCE_RATIO:
            status, verdict = "supported", "evidence supports the claim"
        elif contradiction_mass >= support_mass * DOMINANCE_RATIO:
            status, verdict = "contradicted", "evidence contradicts the claim"
        else:
            status, verdict = "mixed", "claims have mixed evidence"
    elif support_ok:
        status, verdict = "supported", "evidence supports the claim"
    elif contradiction_ok:
        status, verdict = "contradicted", "evidence contradicts the claim"
    else:
        status, verdict = "insufficient_evidence", "insufficient evidence"

    return {
        "support": round(avg_support, 4),
        "contradiction": round(avg_contradiction, 4),
        "net": round(net, 4),
        "status": status,
        "verdict": verdict,
        "nli_available": True,
        "evidence_count": len(nli_results),
        "evidence_used_count": len(nli_results),
        "supporting_count": len(supporting),
        "contradicting_count": len(contradicting),
        "neutral_count": len(neutral),
        "independent_supporting": independent_supporting,
        "independent_contradicting": independent_contradicting,
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
