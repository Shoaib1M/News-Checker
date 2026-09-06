"""What users paste, versus the proposition they are asking about.

WHY THIS EXISTS:
Every stage reads the claim — triage, entity extraction, query generation, and
NLI, which uses it as the hypothesis. All of them were reading the raw
submission, and people do not submit propositions. They submit what they saw,
with the framing they saw it in. Measured on the real stages:

    "is it true that the prime minister of india resigned?"
        triaged not_a_claim, search_worthwhile=False — never searched at all.
        The most natural way there is to ask a fact-checker a question,
        answered with "no verifiable claim found".

    "https://twitter.com/x/status/123 The prime minister of India resigned"
        primary entities ['India', 'Twitter']; a dispatched query was
        "India Twitter resigned".

    '"Google banned in US" - Reuters, March 2024'
        primary entities ['Google', 'Reuters', 'March', 'United States'];
        a dispatched query was "Google Reuters banned".

    "🚨🚨 GOOGLE BANNED IN ALL US CITIES 🚨🚨 #breaking #news"
        the first dispatched query is the submission itself, so the emoji and
        hashtags went verbatim to a news index.

The risk runs the other way too, and the negative cases below are the point of
the file: a normaliser that trims one word too many answers a different
question from the one asked.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from claim_normalizer import normalize_claim  # noqa: E402
from claim_triage import triage_claim  # noqa: E402
from claim_decomposer import decompose_claim  # noqa: E402
from query_generator import QueryGenerator  # noqa: E402


class TestPackagingIsRemoved(unittest.TestCase):

    def test_a_wire_label_goes(self):
        self.assertEqual(
            normalize_claim("BREAKING: The United States has banned Google"),
            "The United States has banned Google")

    def test_a_pasted_url_goes(self):
        self.assertEqual(
            normalize_claim("https://twitter.com/x/status/123 The PM of India resigned"),
            "The PM of India resigned")

    def test_hashtags_and_handles_go(self):
        self.assertEqual(
            normalize_claim("The PM of India resigned #breaking @reuters"),
            "The PM of India resigned")

    def test_emoji_go_without_touching_letters(self):
        self.assertEqual(
            normalize_claim("🚨 GOOGLE BANNED IN ALL US CITIES 🚨"),
            "GOOGLE BANNED IN ALL US CITIES")

    def test_a_quoted_headline_with_an_attribution(self):
        self.assertEqual(
            normalize_claim('"Google banned in US" - Reuters, March 2024'),
            "Google banned in US")

    def test_repeated_emphasis_punctuation_is_collapsed(self):
        self.assertEqual(
            normalize_claim("Google got banned in the US???"),
            "Google got banned in the US")

    def test_conversational_framing_goes(self):
        for submission, expected in (
            ("is it true that the prime minister of india resigned?",
             "the prime minister of india resigned"),
            ("so i heard that google got banned in the us, can someone confirm",
             "google got banned in the us"),
            ("Apparently the central bank raised interest rates. Is this true?",
             "the central bank raised interest rates."),
            ("Please fact-check that Apple announced a foldable iPhone",
             "Apple announced a foldable iPhone"),
        ):
            with self.subTest(submission=submission):
                self.assertEqual(normalize_claim(submission), expected)


class TestTheClaimItselfIsNeverChanged(unittest.TestCase):
    """A normaliser that trims one word too many answers a different question."""

    UNCHANGED = [
        "The prime minister of India resigned",
        "It is true that a triangle has three sides",   # assertion, not a question
        "The United States did not ban Google",         # negation
        "All swans are white",                          # quantifier
        "The vaccine may be 95% effective",             # hedge
        "Only three ministers resigned",
        "Why did the prime minister resign?",           # a real question
        "Reuters reported that the PM resigned",        # attribution IS the claim
    ]

    def test_these_pass_through_untouched(self):
        for claim in self.UNCHANGED:
            with self.subTest(claim=claim):
                self.assertEqual(normalize_claim(claim), claim)

    def test_a_submission_that_is_only_packaging_is_returned_as_it_was(self):
        """Triage must see it to say there is nothing here."""
        for junk in ("🚨🚨🚨", "#breaking", "https://example.com/story"):
            with self.subTest(junk=junk):
                self.assertEqual(normalize_claim(junk), junk)

    def test_empty_input_is_survivable(self):
        self.assertEqual(normalize_claim(""), "")
        self.assertEqual(normalize_claim("   "), "")


class TestTheEffectOnTheStagesDownstream(unittest.TestCase):
    """The reason any of this matters."""

    def test_a_question_form_claim_becomes_checkable(self):
        raw = "is it true that the prime minister of india resigned?"
        self.assertFalse(triage_claim(raw).search_worthwhile,
                         "the premise: this was never searched")
        self.assertTrue(triage_claim(normalize_claim(raw)).search_worthwhile)

    def test_a_genuine_question_is_still_refused(self):
        """The guard on the fix above: not every "?" is a claim in disguise."""
        raw = "Why did the prime minister resign?"
        self.assertEqual(triage_claim(normalize_claim(raw)).kind, "not_a_claim")

    def test_a_pasted_url_stops_becoming_an_entity(self):
        raw = "https://twitter.com/x/status/123 The prime minister of India resigned"
        self.assertIn("Twitter", decompose_claim(raw).primary_entities)
        self.assertNotIn("Twitter",
                         decompose_claim(normalize_claim(raw)).primary_entities)

    def test_an_attribution_stops_becoming_an_entity(self):
        raw = '"Google banned in US" - Reuters, March 2024'
        entities = decompose_claim(raw).primary_entities
        self.assertIn("Reuters", entities)
        self.assertIn("March", entities)
        cleaned = decompose_claim(normalize_claim(raw)).primary_entities
        self.assertNotIn("Reuters", cleaned)
        self.assertNotIn("March", cleaned)

    def test_no_dispatched_query_carries_emoji_or_hashtags(self):
        raw = "🚨🚨 GOOGLE BANNED IN ALL US CITIES 🚨🚨 #breaking #news"
        generator = QueryGenerator()
        queries = [q["query"]
                   for q in generator.generate_queries(normalize_claim(raw))[:4]]
        for query in queries:
            with self.subTest(query=query):
                self.assertNotIn("#", query)
                self.assertNotIn("🚨", query)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── Through the API, where it has to hold together ───────────────────

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import evidence_pipeline  # noqa: E402
import main  # noqa: E402
from providers import ProviderDiagnostic  # noqa: E402


class _NoNLI:
    is_available = False
    is_ready = False
    status = {"status": "disabled", "enabled": False, "model": "stub", "error": None}

    def score_many(self, claim, passages):
        return [{"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0,
                 "available": False} for _ in passages]


class TestThroughTheApi(unittest.TestCase):
    """Retrieval is stubbed empty: what is asserted is whether a search was
    ATTEMPTED, and what the response echoes back to the user."""

    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(main.app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def check(self, statement):
        nli = _NoNLI()
        with patch.object(
            evidence_pipeline, "search_all_providers",
            lambda q, **k: ([], [ProviderDiagnostic(
                provider="google_news", query=q, enabled=True,
                status="no_results")])
        ), patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
                patch.object(main, "get_nli_service", lambda: nli):
            response = self.client.post("/api/check", json={"statement": statement})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_the_question_form_is_no_longer_refused(self):
        body = self.check("is it true that the prime minister of india resigned?")
        self.assertNotEqual(
            body["verification"]["status"], "not_a_claim",
            "the most natural way to ask a fact-checker a question",
        )

    def test_a_submission_full_of_emoji_is_still_checked(self):
        body = self.check("🚨🚨 GOOGLE BANNED IN ALL US CITIES 🚨🚨 #breaking")
        self.assertNotEqual(body["verification"]["status"], "not_a_claim")

    def test_a_genuine_question_is_still_refused(self):
        body = self.check("Why did the prime minister of India resign?")
        self.assertEqual(body["verification"]["status"], "not_a_claim")

    def test_the_response_echoes_the_user_s_own_words(self):
        """The UI shows this and history stores it. Normalising it would show
        the user something they did not write."""
        for submission in (
            "is it true that the prime minister of india resigned?",
            "🚨🚨 GOOGLE BANNED IN ALL US CITIES 🚨🚨 #breaking",
            "BREAKING: The United States has banned Google!!!",
        ):
            with self.subTest(submission=submission):
                self.assertEqual(self.check(submission)["statement"], submission)
