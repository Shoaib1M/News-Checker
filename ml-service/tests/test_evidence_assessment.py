"""Tests for evidence aggregation (evidence_aggregator.py).

These replace the old tests that exercised evidence_scraper.py, which was
dead code (superseded by evidence_pipeline.py + evidence_aggregator.py and
never called from main.py).
"""

import sys
from pathlib import Path
import unittest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from evidence_aggregator import ClassifiedEvidence, compute_stance, count_independent_groups


def _evidence(
    url,
    source_tier="reporting",
    source_weight=0.8,
    support=0.0,
    contradiction=0.0,
    nli_available=True,
    stance="unclear",
    source="Example",
):
    return ClassifiedEvidence(
        url=url,
        source=source,
        source_tier=source_tier,
        source_weight=source_weight,
        support_score=support,
        contradiction_score=contradiction,
        nli_available=nli_available,
        stance=stance,
    )


class ComputeStanceTests(unittest.TestCase):
    def test_no_results_is_insufficient_evidence(self):
        summary = compute_stance([])
        self.assertEqual(summary["status"], "insufficient_evidence")
        self.assertFalse(summary["nli_available"])

    def test_unavailable_nli_cannot_create_a_verdict(self):
        """A candidate that never got NLI-classified must not count as evidence."""
        results = [_evidence("https://www.cdc.gov/report", nli_available=False)]
        summary = compute_stance(results)
        self.assertEqual(summary["status"], "insufficient_evidence")
        self.assertEqual(summary["evidence_count"], 0)

    def test_primary_source_has_more_weight_than_repeated_reporting(self):
        results = [
            _evidence(
                "https://primary.example/story", source_tier="primary",
                source_weight=1.0, support=0.90, contradiction=0.05, stance="supports",
            ),
            _evidence(
                "https://reporting-a.example/story", source_tier="reporting",
                source_weight=0.8, support=0.05, contradiction=0.90, stance="contradicts",
            ),
            _evidence(
                "https://reporting-b.example/story", source_tier="reporting",
                source_weight=0.8, support=0.05, contradiction=0.90, stance="contradicts",
            ),
        ]
        summary = compute_stance(results)
        self.assertEqual(summary["status"], "mixed")

    def test_consistent_support_produces_supported_status(self):
        results = [
            _evidence(
                "https://reuters.com/a", source_tier="reporting", source_weight=0.8,
                support=0.80, contradiction=0.05, stance="supports",
            ),
            _evidence(
                "https://apnews.com/b", source_tier="reporting", source_weight=0.8,
                support=0.75, contradiction=0.05, stance="supports",
            ),
        ]
        summary = compute_stance(results)
        self.assertEqual(summary["status"], "supported")
        self.assertGreater(summary["net"], 0)

    def test_support_and_contradiction_together_is_mixed(self):
        results = [
            _evidence("https://a.example/x", support=0.80, contradiction=0.05, stance="supports"),
            _evidence("https://b.example/y", support=0.05, contradiction=0.80, stance="contradicts"),
        ]
        summary = compute_stance(results)
        self.assertEqual(summary["status"], "mixed")


class IndependentGroupsTests(unittest.TestCase):
    def test_same_domain_counts_once(self):
        results = [
            _evidence("https://www.reuters.com/a", nli_available=True),
            _evidence("https://reuters.com/b", nli_available=True),
        ]
        self.assertEqual(count_independent_groups(results), 1)

    def test_distinct_domains_count_separately(self):
        results = [
            _evidence("https://www.reuters.com/a", nli_available=True),
            _evidence("https://apnews.com/b", nli_available=True),
            _evidence("https://bbc.com/c", nli_available=True),
        ]
        self.assertEqual(count_independent_groups(results), 3)

    def test_unclassified_candidates_are_excluded(self):
        results = [
            _evidence("https://www.reuters.com/a", nli_available=True),
            _evidence("https://apnews.com/b", nli_available=False),
        ]
        self.assertEqual(count_independent_groups(results), 1)


if __name__ == "__main__":
    unittest.main()
