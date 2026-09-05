"""Deduplication and entity matching — two places that silently lose evidence.

WHY THIS EXISTS:
Both stages here decide what the rest of the pipeline is even allowed to see,
and both had failures that are invisible downstream:

  - Deduplication merged two headlines that differ only by "not". For a
    fact-checker that is the worst possible thing to discard, because
    surfacing contradictions is the entire job.
  - Entity matching extracted entities from the claim and from the document
    with two different extractors that knew different things, so it failed in
    both directions at once — matching the English pronoun "us" against the
    United States, while failing to match a lowercase "google" headline
    against a claim about Google.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from claim_decomposer import decompose_claim  # noqa: E402
from providers import SearchResult  # noqa: E402
from providers.registry import deduplicate  # noqa: E402
from relevance_filter import RelevanceFilter  # noqa: E402


def result(url: str, title: str) -> SearchResult:
    return SearchResult(url=url, title=title, snippet="", text="", provider="p", source="")


class TestDeduplicationKeepsDisagreement(unittest.TestCase):
    """Near-identical headlines that say opposite things are not duplicates."""

    def test_a_negation_makes_two_headlines_distinct(self):
        kept = deduplicate([
            result("https://a.example/1",
                   "Court rules Google must be banned in all US cities immediately"),
            result("https://b.example/2",
                   "Court rules Google must not be banned in all US cities immediately"),
        ])
        self.assertEqual(
            len(kept), 2,
            "headlines differing by 'not' are 0.92 Jaccard-similar; merging them "
            "silently discards the contradicting story",
        )

    def test_opposite_events_make_two_headlines_distinct(self):
        kept = deduplicate([
            result("https://a.example/1",
                   "Regulators approve the merger between the two large companies"),
            result("https://b.example/2",
                   "Regulators reject the merger between the two large companies"),
        ])
        self.assertEqual(len(kept), 2)

    def test_a_denial_makes_two_headlines_distinct(self):
        kept = deduplicate([
            result("https://a.example/1",
                   "Minister confirms the resignation reports circulating this week"),
            result("https://b.example/2",
                   "Minister denies the resignation reports circulating this week"),
        ])
        self.assertEqual(len(kept), 2)

    def test_genuine_duplicates_are_still_merged(self):
        kept = deduplicate([
            result("https://a.example/1", "India PM resigns after coalition talks collapse in Delhi"),
            result("https://b.example/2", "India PM resigns after coalition talks collapse in Delhi"),
        ])
        self.assertEqual(len(kept), 1)

    def test_the_same_url_is_never_kept_twice(self):
        kept = deduplicate([
            result("https://a.example/1", "One headline"),
            result("https://a.example/1", "A different headline, same URL"),
        ])
        self.assertEqual(len(kept), 1)


class TestEntityMatching(unittest.TestCase):

    CLAIM = "The United States banned Google across all its cities"

    def setUp(self):
        self.filter = RelevanceFilter()

    def score(self, title, body):
        return self.filter.assess_document_relevance(self.CLAIM, title, body, body * 10)

    def test_the_pronoun_us_is_not_the_united_states(self):
        score = self.score(
            "Google blocked our account and never told us why",
            "Google blocked us from the service without warning. They never told us the reason.",
        )
        self.assertLess(
            score.entity_match_score, 1.0,
            "lowercase 'us' is the pronoun and appears in most English prose",
        )

    def test_a_capitalised_abbreviation_is_the_country(self):
        score = self.score(
            "US lawmakers move to ban Google services nationwide",
            "A bill before the U.S. Congress would ban Google across all US cities.",
        )
        self.assertEqual(score.entity_match_score, 1.0)

    def test_a_lowercase_headline_still_matches_its_entities(self):
        """The document-side extractor did not know lowercase org names."""
        score = self.score(
            "united states lawmakers move to ban google services",
            "A bill in the united states would ban google nationwide.",
        )
        self.assertEqual(score.entity_match_score, 1.0)

    def test_america_counts_as_the_united_states(self):
        score = self.score(
            "America moves to ban Google services nationwide",
            "A bill in America would ban Google across all cities.",
        )
        self.assertEqual(score.entity_match_score, 1.0)

    def test_a_different_country_does_not_score_a_full_match(self):
        score = self.score(
            "France moves to ban Google services nationwide",
            "A bill in France would ban Google across all French cities.",
        )
        self.assertLess(score.entity_match_score, 1.0)


class TestClaimEntityCanonicalisation(unittest.TestCase):

    def test_the_bare_abbreviation_resolves_to_the_country(self):
        entities = decompose_claim("US bans Google nationwide").primary_entities
        self.assertIn("United States", entities)
        self.assertNotIn("Us", entities)

    def test_the_full_name_is_not_duplicated_by_its_abbreviation(self):
        entities = decompose_claim("The United States and the US both act").primary_entities
        self.assertEqual(
            [e for e in entities if e.lower() == "united states"], ["United States"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
