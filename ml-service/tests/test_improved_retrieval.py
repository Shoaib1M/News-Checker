"""Assertion-based tests for claim decomposition, query generation, and
relevance filtering — the pipeline stages that run before NLI.

These replace the old print-only demo script of the same name, which never
asserted anything and imported the now-deleted evidence_scraper module.
"""

import sys
from pathlib import Path
import unittest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from claim_decomposer import decompose_claim
from query_generator import QueryGenerator
from relevance_filter import RelevanceFilter


class ClaimDecompositionTests(unittest.TestCase):
    def test_future_temporal_claim(self):
        decomp = decompose_claim("The name of united states is being changed to india by 2050.")
        self.assertEqual(decomp.temporal_constraints, "future")
        self.assertIn("United States", decomp.primary_entities)
        self.assertIn("India", decomp.primary_entities)

    def test_negation_and_attribution_on_denial(self):
        """'NASA denies X' must be flagged as a negated, attributed statement —
        this is what lets the system tell it apart from 'NASA says X'."""
        decomp = decompose_claim("NASA denies the asteroid will pass close to Earth.")
        self.assertTrue(decomp.negation)
        self.assertEqual(decomp.attribution, "denies")

    def test_affirmative_attribution_has_no_negation(self):
        decomp = decompose_claim("NASA says the asteroid will pass close to Earth.")
        self.assertFalse(decomp.negation)
        self.assertEqual(decomp.attribution, "says")

    def test_speculative_modality(self):
        decomp = decompose_claim("NASA says the asteroid might hit Earth.")
        self.assertEqual(decomp.modality, "speculative")

    def test_factual_claim_has_no_negation(self):
        decomp = decompose_claim("Water freezes at 0 degrees C at sea level.")
        self.assertFalse(decomp.negation)
        self.assertEqual(decomp.modality, "factual")

    def test_explicit_negation_word(self):
        decomp = decompose_claim("Officials confirmed the policy was not implemented.")
        self.assertTrue(decomp.negation)


class QueryGenerationTests(unittest.TestCase):
    def setUp(self):
        self.generator = QueryGenerator()

    def test_generates_multiple_typed_queries(self):
        claim = "The name of united states is being changed to india by 2050."
        queries = self.generator.generate_queries(claim)
        self.assertGreater(len(queries), 3)
        purposes = {q["purpose"] for q in queries}
        self.assertIn("exact_claim", purposes)
        self.assertIn("proposition", purposes)
        # Two-entity claims should get an explicit entity-relationship query,
        # not just a bag of separate keyword queries.
        self.assertIn("entity_relationship", purposes)

    def test_exact_claim_query_preserves_wording(self):
        claim = "Water freezes at 0 degrees C at sea level."
        queries = self.generator.generate_queries(claim)
        exact = next(q for q in queries if q["purpose"] == "exact_claim")
        self.assertIn("Water freezes", exact["query"])


class RelevanceFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = RelevanceFilter()
        self.claim = "The name of united states is being changed to india by 2050."
        self.documents = [
            {
                "title": "Kennedy Center reportedly changed rules before vote to add Trump's name",
                "snippet": "The Kennedy Center made changes to voting procedures...",
                "text": "The Kennedy Center reportedly changed its rules...",
            },
            {
                "title": "A job that changed me: Being a theatre usher cracked open my heart to beauty",
                "snippet": "Working as a theatre usher changed my perspective on life...",
                "text": "I worked as a theatre usher and it changed everything...",
            },
            {
                "title": "Airline industry chiefs say 2050 net zero goal now unlikely",
                "snippet": "Industry leaders gathered to discuss 2050 climate targets...",
                "text": "The airline industry is concerned about meeting 2050 net zero goals...",
            },
            {
                "title": "Geopolitical tensions as India and US relations shift",
                "snippet": "The relationship between India and the United States continues to evolve...",
                "text": "Relations between India and the United States have been changing significantly...",
            },
        ]

    def test_keyword_collision_articles_are_excluded(self):
        """Sharing only 'changed' or '2050' with the claim must not qualify —
        this is the exact false-positive class the relevance filter exists to stop."""
        included, excluded = self.filter.filter_documents(self.claim, self.documents, strict=True)
        included_titles = {doc["title"] for doc in included}
        self.assertNotIn(
            "Kennedy Center reportedly changed rules before vote to add Trump's name",
            included_titles,
        )
        self.assertNotIn(
            "A job that changed me: Being a theatre usher cracked open my heart to beauty",
            included_titles,
        )
        self.assertNotIn(
            "Airline industry chiefs say 2050 net zero goal now unlikely",
            included_titles,
        )

    def test_genuinely_relevant_article_is_included(self):
        included, _ = self.filter.filter_documents(self.claim, self.documents, strict=True)
        included_titles = {doc["title"] for doc in included}
        self.assertIn("Geopolitical tensions as India and US relations shift", included_titles)

    def test_ban_claim_is_not_matched_by_unrelated_domains(self):
        """Regression for 'US government considers banning Google': a bond-market
        article and a Google-advertising article must not pass as relevant."""
        claim = "The US government is considering banning Google."
        documents = [
            {
                "title": "US bond markets rattled by inflation data",
                "snippet": "Treasury yields rose sharply this week...",
                "text": "Bond markets reacted to new inflation figures released by the Fed...",
            },
            {
                "title": "Google expands advertising tools for small businesses",
                "snippet": "New ad formats roll out globally...",
                "text": "Google announced new advertising products aimed at small businesses...",
            },
        ]
        included, _ = self.filter.filter_documents(claim, documents, strict=True)
        self.assertEqual(included, [])


if __name__ == "__main__":
    unittest.main()
