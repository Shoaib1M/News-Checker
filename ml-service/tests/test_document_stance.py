"""Per-document stance — turning passage scores into "what this article says".

WHY THIS EXISTS:
This stage had the most damaging bug in the codebase, because it inverted the
best sources rather than merely ignoring them.

A debunking article is built out of the claim it refutes. It quotes it —
"Posts claim the United States banned Google in all its cities" — and an NLI
model scores that as strongly entailing the claim, because the claim is
literally in the sentence. Then it refutes it: "This is false. No such ban
exists."

The old code found the single passage with the highest max(entail, contradict)
and read BOTH scores off that one passage. The quote scored 0.88; the
refutation 0.80. So the quote won, its near-zero contradiction score was read
off with it, and a PolitiFact article debunking the claim was recorded as
supporting it — at 0.95 source weight, the highest in the system.

Three changes, each pinned below:
  1. Strongest entailment and strongest contradiction are found independently.
  2. Passages that merely REPORT a claim are excluded from the entailment
     maximum. Ordinary attribution ("officials said") is untouched.
  3. When both directions clear the threshold, one must dominate; otherwise
     the document is arguing both ways and "unclear" is the honest label.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import evidence_pipeline  # noqa: E402
from evidence_pipeline import (  # noqa: E402
    STANCE_DOMINANCE,
    STANCE_THRESHOLD,
    _CLAIM_REPORTING_FRAME,
    run_pipeline,
)
from providers import ProviderDiagnostic, SearchResult  # noqa: E402


DIAGNOSTIC = [ProviderDiagnostic(
    provider="p", query="q", enabled=True, status="success",
    raw_result_count=1, new_result_count=1,
)]


class ScriptedNLI:
    """Scores passages by a substring -> (entailment, contradiction) table."""

    is_available = True

    def __init__(self, table):
        self.table = table

    def score_many(self, claim, passages):
        scores = []
        for passage in passages:
            lowered = passage.lower()
            entail, contradict = 0.04, 0.03
            for needle, (e, c) in self.table.items():
                if needle in lowered:
                    entail, contradict = e, c
                    break
            scores.append({
                "entailment": entail, "contradiction": contradict,
                "neutral": max(0.0, 1 - entail - contradict), "available": True,
            })
        return scores


def stance_of(claim, title, snippet, body, table, domain="politifact.com"):
    """Run one document through the pipeline and return its EvidenceResult."""
    results = [SearchResult(url=f"https://{domain}/story", title=title,
                            snippet=snippet, text=body * 8, provider="p", source=domain)]
    with patch.object(evidence_pipeline, "search_all_providers",
                      lambda q, **k: (results, DIAGNOSTIC)), \
         patch.object(evidence_pipeline, "get_nli_service", lambda: ScriptedNLI(table)):
        outcome = run_pipeline(claim)
    return outcome


CLAIM = "The United States banned Google across all its cities"

FACT_CHECK_BODY = (
    "Fact check: No, the US has not banned Google. "
    "Posts claim the United States banned Google in all its cities. "
    "This is false. No such ban exists and no bill has been introduced. "
)

FACT_CHECK_SCORES = {
    "posts claim": (0.88, 0.03),      # the quoted claim
    "this is false": (0.05, 0.80),    # the refutation
    "has not banned": (0.10, 0.72),   # the headline
}


class TestFactChecksAreNotInverted(unittest.TestCase):

    def test_a_debunking_article_contradicts_the_claim_it_debunks(self):
        outcome = stance_of(
            CLAIM, "Fact check: No, the US has not banned Google",
            "Posts claim the United States banned Google in all its cities.",
            FACT_CHECK_BODY, FACT_CHECK_SCORES,
        )
        self.assertEqual(len(outcome.evidence), 1)
        self.assertEqual(outcome.evidence[0].stance, "contradicts")

    def test_the_quoted_claim_does_not_become_the_support_score(self):
        outcome = stance_of(
            CLAIM, "Fact check: No, the US has not banned Google",
            "Posts claim the United States banned Google in all its cities.",
            FACT_CHECK_BODY, FACT_CHECK_SCORES,
        )
        self.assertLess(outcome.evidence[0].support_score, STANCE_THRESHOLD)

    def test_the_overall_verdict_follows(self):
        outcome = stance_of(
            CLAIM, "Fact check: No, the US has not banned Google",
            "Posts claim the United States banned Google in all its cities.",
            FACT_CHECK_BODY, FACT_CHECK_SCORES,
        )
        self.assertEqual(outcome.stance["status"], "contradicted")


class TestOrdinaryReportingIsUnaffected(unittest.TestCase):
    """The narrow point of the frame regex: normal journalism must still count."""

    def test_officials_said_is_not_a_claim_reporting_frame(self):
        self.assertIsNone(_CLAIM_REPORTING_FRAME.search(
            "Officials said the prime minister resigned on Tuesday"
        ))

    def test_according_to_is_not_a_claim_reporting_frame(self):
        self.assertIsNone(_CLAIM_REPORTING_FRAME.search(
            "According to two people familiar with the matter, the deal closed"
        ))

    def test_posts_claim_is_a_claim_reporting_frame(self):
        self.assertIsNotNone(_CLAIM_REPORTING_FRAME.search(
            "Posts claim the United States banned Google"
        ))

    def test_viral_and_debunked_are_claim_reporting_frames(self):
        for text in (
            "A viral video appeared to show the announcement",
            "The widely shared image has been debunked",
            "Social media users shared the screenshot",
        ):
            with self.subTest(text=text):
                self.assertIsNotNone(_CLAIM_REPORTING_FRAME.search(text))

    def test_attributed_reporting_still_supports_a_claim(self):
        outcome = stance_of(
            "The prime minister of India resigned this morning",
            "Officials said the prime minister resigned on Tuesday",
            "The prime minister resigned on Tuesday, officials said.",
            "Officials said the prime minister resigned on Tuesday. "
            "The prime minister resigned after coalition talks failed. ",
            {"resigned": (0.91, 0.03)},
            domain="reuters.com",
        )
        self.assertEqual(outcome.evidence[0].stance, "supports")


class TestBothDirectionsRequireDominance(unittest.TestCase):

    def test_a_document_arguing_both_ways_is_unclear(self):
        outcome = stance_of(
            CLAIM, "Analysts split on the proposed Google restrictions",
            "Some argue a ban is already in force; others say no such rule exists.",
            "Some argue a ban is already in force in every city. "
            "Others say no such rule exists anywhere in the country. ",
            {"already in force": (0.70, 0.05), "no such rule": (0.05, 0.62)},
            domain="reuters.com",
        )
        self.assertEqual(outcome.evidence[0].stance, "unclear")

    def test_a_dominant_direction_still_wins(self):
        outcome = stance_of(
            CLAIM, "Google banned across all US cities under new rule",
            "The ban takes effect immediately in every city.",
            "The ban takes effect immediately in every city nationwide. "
            "One trade group weakly disputed the scope of the rule. ",
            {"takes effect": (0.92, 0.03), "weakly disputed": (0.05, 0.40)},
            domain="reuters.com",
        )
        self.assertEqual(outcome.evidence[0].stance, "supports")

    def test_the_dominance_ratio_is_what_separates_them(self):
        """Documents the constants rather than re-deriving them."""
        self.assertGreater(STANCE_DOMINANCE, 1.0)
        self.assertLess(0.62 * STANCE_DOMINANCE, 0.70 * STANCE_DOMINANCE)
        self.assertLess(0.70, 0.62 * STANCE_DOMINANCE)   # 0.70 vs 0.62 -> unclear
        self.assertGreater(0.92, 0.40 * STANCE_DOMINANCE)  # 0.92 vs 0.40 -> supports


if __name__ == "__main__":
    unittest.main(verbosity=2)
