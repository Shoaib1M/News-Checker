"""Stance aggregation — how classified sources become a direction.

WHY THIS EXISTS:
This is where "what the sources say" turns into "what we tell the user", and
it had three defects that each produce a confidently wrong answer:

  1. Direction scores were means over *every* classified source, neutrals
     included, so adding on-topic articles that said nothing either way could
     flip a supported verdict to insufficient. The verdict was not monotonic
     in the evidence.
  2. "Mixed" fired on raw counts, so one weak dissent from an unclassified
     blog outweighed five strong reports from reputable outlets.
  3. Independence was documented but never applied: four copies of one wire
     story from one newsroom counted as four confirmations and earned high
     confidence.

Each is pinned below with the shape that exposed it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from evidence_aggregator import ClassifiedEvidence, compute_stance  # noqa: E402
from main import _compute_confidence, _independent_backing  # noqa: E402


def ev(domain, support, contradiction, stance, tier="reporting", weight=0.8):
    return ClassifiedEvidence(
        url=f"https://{domain}/story", publisher=domain, source=domain,
        source_tier=tier, source_weight=weight, support_score=support,
        contradiction_score=contradiction, nli_available=True, stance=stance,
    )


SUPPORT = lambda d, s=0.9: ev(d, s, 0.02, "supports")          # noqa: E731
CONTRA = lambda d, c=0.9: ev(d, 0.02, c, "contradicts")        # noqa: E731
NEUTRAL = lambda d: ev(d, 0.04, 0.03, "unclear")               # noqa: E731


class TestNeutralsDoNotDilute(unittest.TestCase):
    """The verdict must be monotonic: more evidence can't mean less certainty."""

    def test_neutral_coverage_does_not_overturn_a_clear_entailment(self):
        alone = compute_stance([SUPPORT("reuters.com", 0.93)])
        with_neutrals = compute_stance([
            SUPPORT("reuters.com", 0.93),
            NEUTRAL("bbc.com"), NEUTRAL("npr.org"), NEUTRAL("wsj.com"),
        ])
        self.assertEqual(alone["status"], "supported")
        self.assertEqual(
            with_neutrals["status"], "supported",
            "adding articles that say nothing either way must not flip the verdict",
        )

    def test_the_direction_score_reflects_only_the_sources_taking_it(self):
        stance = compute_stance([
            SUPPORT("reuters.com", 0.93), NEUTRAL("bbc.com"), NEUTRAL("npr.org"),
        ])
        self.assertAlmostEqual(stance["support"], 0.93, places=2)

    def test_neutrals_are_still_counted_and_reported(self):
        stance = compute_stance([SUPPORT("reuters.com"), NEUTRAL("bbc.com")])
        self.assertEqual(stance["neutral_count"], 1)
        self.assertEqual(stance["evidence_count"], 2)

    def test_neutral_only_evidence_yields_no_direction(self):
        stance = compute_stance([NEUTRAL("bbc.com"), NEUTRAL("npr.org")])
        self.assertEqual(stance["status"], "insufficient_evidence")


class TestMixedRequiresAGenuineContest(unittest.TestCase):

    def test_a_lone_weak_dissent_does_not_outweigh_a_consensus(self):
        stance = compute_stance(
            [SUPPORT(f"outlet{i}.com") for i in range(5)]
            + [ev("randomblog.com", 0.02, 0.40, "contradicts", tier="unclassified", weight=0.0)]
        )
        self.assertEqual(stance["status"], "supported")

    def test_a_real_disagreement_is_reported_as_mixed(self):
        stance = compute_stance([SUPPORT("bbc.com", 0.85), CONTRA("wsj.com", 0.80)])
        self.assertEqual(stance["status"], "mixed")

    def test_a_strong_fact_check_can_outweigh_weaker_reporting(self):
        """Source tier is what lets one authoritative correction win."""
        stance = compute_stance([
            ev("aggregator1.com", 0.50, 0.02, "supports", tier="unclassified", weight=0.0),
            ev("politifact.com", 0.02, 0.95, "contradicts", tier="fact-check", weight=0.95),
        ])
        self.assertEqual(stance["status"], "contradicted")

    def test_weak_scores_on_both_sides_produce_no_verdict(self):
        stance = compute_stance([
            ev("a.com", 0.20, 0.02, "supports"), ev("b.com", 0.02, 0.18, "contradicts"),
        ])
        self.assertEqual(stance["status"], "insufficient_evidence")


class TestSyndicationIsNotConfirmation(unittest.TestCase):
    """Ten copies of one story are one story."""

    def test_repeats_from_one_publisher_count_as_one_independent_source(self):
        stance = compute_stance([SUPPORT("reuters.com") for _ in range(4)])
        self.assertEqual(stance["evidence_count"], 4)
        self.assertEqual(stance["independent_supporting"], 1)

    def test_distinct_publishers_are_counted_separately(self):
        stance = compute_stance([SUPPORT("reuters.com"), SUPPORT("bbc.com"), SUPPORT("npr.org")])
        self.assertEqual(stance["independent_supporting"], 3)

    def test_confidence_follows_independent_publishers_not_article_count(self):
        syndicated = compute_stance([SUPPORT("reuters.com") for _ in range(4)])
        distinct = compute_stance([SUPPORT("reuters.com"), SUPPORT("bbc.com"), SUPPORT("npr.org")])
        self.assertEqual(
            _compute_confidence(syndicated["status"], _independent_backing(syndicated), True),
            "low",
        )
        self.assertEqual(
            _compute_confidence(distinct["status"], _independent_backing(distinct), True),
            "high",
        )

    def test_www_and_bare_host_are_the_same_publisher(self):
        stance = compute_stance([
            ClassifiedEvidence(url="https://www.reuters.com/a", publisher="", source="",
                               source_tier="reporting", source_weight=0.8, support_score=0.9,
                               contradiction_score=0.02, nli_available=True, stance="supports"),
            ClassifiedEvidence(url="https://reuters.com/b", publisher="", source="",
                               source_tier="reporting", source_weight=0.8, support_score=0.9,
                               contradiction_score=0.02, nli_available=True, stance="supports"),
        ])
        self.assertEqual(stance["independent_supporting"], 1)


class TestConfidenceDirection(unittest.TestCase):

    def test_a_contradicted_verdict_counts_the_contradicting_publishers(self):
        stance = compute_stance([CONTRA("a.com"), CONTRA("b.com"), CONTRA("c.com"), SUPPORT("d.com", 0.4)])
        self.assertEqual(stance["status"], "contradicted")
        self.assertEqual(_independent_backing(stance), 3)

    def test_unclassified_evidence_is_never_confident(self):
        self.assertEqual(_compute_confidence("supported", 5, nli_available=False), "low")


if __name__ == "__main__":
    unittest.main(verbosity=2)
