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


class TestCredibleSourcesReachNLI(unittest.TestCase):
    """Which candidates get spent on NLI, when there are more than fit.

    Only `max_results` documents are ever classified, and they were chosen by
    lexical relevance alone. For a viral false claim that is backwards: the
    posts repeating the claim use its precise wording, while the debunkings
    describe the situation in their own. Measured on the pool below, a
    PolitiFact fact-check ranked NINTH behind eight rumour blogs and never
    reached NLI — so the system would have classified eight copies of the
    rumour and reported the claim supported.
    """

    CLAIM = "The United States banned Google across all its cities"

    RUMOUR_BLOGS = [
        ("Google ban rumours spread across all US cities",
         "Reports of a Google ban in all US cities spread widely."),
        ("Google banned in all United States cities, users say",
         "A Google ban across all US cities was reported."),
        ("US cities Google ban: everything we know",
         "The Google ban across all US cities explained."),
        ("All US cities affected by Google ban claims",
         "The claimed Google ban across all cities."),
        ("Google ban across every US city discussed",
         "Discussion of a Google ban across US cities."),
        ("US Google ban in all cities trends online",
         "The Google ban in all US cities trended."),
        ("Google ban all US cities update",
         "Update on the Google ban across all US cities."),
        ("Google banned across US cities, posts allege",
         "Posts allege Google was banned across US cities."),
    ]

    CREDIBLE = [
        ("politifact.com", "Fact check: the US has not banned Google",
         "No such nationwide prohibition exists, officials confirm."),
        ("reuters.com", "No US ban on Google, regulators confirm",
         "Regulators said no prohibition is in force."),
    ]

    def select(self, max_results=8):
        from claim_verifier import classify_source
        from evidence_pipeline import _select_for_classification
        from relevance_filter import RelevanceFilter

        rows = [(f"blog{i}.example", t, b) for i, (t, b) in enumerate(self.RUMOUR_BLOGS)]
        rows += list(self.CREDIBLE)
        documents = [
            {"url": f"https://{domain}/{i}", "title": title, "snippet": body,
             "text": body * 20, "source": domain, "provider": "p"}
            for i, (domain, title, body) in enumerate(rows)
        ]
        included, _excluded = RelevanceFilter().filter_documents(
            self.CLAIM, documents, strict=True
        )
        selected = _select_for_classification(included, max_results)
        return selected, classify_source

    def test_credible_sources_are_not_crowded_out_by_rumour_posts(self):
        selected, classify_source = self.select()
        tiers = {classify_source(d["url"], d["source"]).tier for d in selected}
        self.assertIn("fact-check", tiers)
        self.assertIn("reporting", tiers)

    def test_the_result_count_is_still_capped(self):
        selected, _ = self.select(max_results=8)
        self.assertEqual(len(selected), 8)

    def test_relevance_order_is_preserved_among_the_chosen(self):
        selected, _ = self.select()
        scores = [d["_relevance_score"].overall_relevance for d in selected]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_with_no_credible_sources_selection_is_plain_relevance_order(self):
        from evidence_pipeline import _select_for_classification
        from relevance_filter import RelevanceFilter

        documents = [
            {"url": f"https://blog{i}.example/{i}", "title": t, "snippet": b,
             "text": b * 20, "source": f"blog{i}.example", "provider": "p"}
            for i, (t, b) in enumerate(self.RUMOUR_BLOGS)
        ]
        included, _ = RelevanceFilter().filter_documents(self.CLAIM, documents, strict=True)
        self.assertEqual(_select_for_classification(included, 4), included[:4])

    def test_a_short_candidate_list_is_returned_untouched(self):
        from evidence_pipeline import _select_for_classification
        documents = [{"url": "https://a.example/1", "source": "a.example"}]
        self.assertEqual(_select_for_classification(documents, 8), documents)


class TestPartialNLIAvailability(unittest.TestCase):
    """Availability is per passage, so it has to be read per passage.

    The check tested only `nli_scores[0]`, which cuts both ways: a document
    classified cleanly seven times over was discarded because its first
    passage happened to fail, and — in the other direction — unavailable
    entries, which report 0.0/0.0/1.0, were allowed into the score comparison
    as though they were real neutral judgements.
    """

    CLAIM = "The prime minister of India resigned this morning"
    BODY = "The prime minister resigned on Tuesday after coalition talks failed. " * 10

    def run_with(self, scorer):
        results = [SearchResult(
            url="https://reuters.com/story", title="India PM resigns",
            snippet="The prime minister resigned.", text=self.BODY,
            provider="p", source="reuters.com",
        )]
        with patch.object(evidence_pipeline, "search_all_providers",
                          lambda q, **k: (results, DIAGNOSTIC)), \
             patch.object(evidence_pipeline, "get_nli_service", lambda: scorer):
            return run_pipeline(self.CLAIM)

    @staticmethod
    def _scorer(available_at):
        class _Partial:
            is_available = True

            def score_many(self, claim, passages):
                return [
                    {"entailment": 0.92, "contradiction": 0.02,
                     "neutral": 0.06, "available": True}
                    if available_at(i) else
                    {"entailment": 0.0, "contradiction": 0.0,
                     "neutral": 1.0, "available": False}
                    for i in range(len(passages))
                ]
        return _Partial()

    def test_a_failure_on_the_first_passage_does_not_discard_the_document(self):
        outcome = self.run_with(self._scorer(lambda i: i != 0))
        self.assertTrue(outcome.evidence[0].nli_available)
        self.assertEqual(outcome.evidence[0].stance, "supports")

    def test_a_failure_on_a_later_passage_does_not_discard_it_either(self):
        outcome = self.run_with(self._scorer(lambda i: i == 0))
        self.assertTrue(outcome.evidence[0].nli_available)
        self.assertEqual(outcome.evidence[0].stance, "supports")

    def test_when_every_passage_fails_the_document_is_unclassified(self):
        outcome = self.run_with(self._scorer(lambda i: False))
        self.assertFalse(outcome.evidence[0].nli_available)
        self.assertEqual(outcome.evidence[0].stance, "unclear")
        self.assertEqual(outcome.stance["status"], "insufficient_evidence")
