"""Recent vs historical coverage: what gets fetched, and what may confirm a claim.

WHY THIS EXISTS:
Entailment has no clock. "India's prime minister resigned on Tuesday" entails
"the prime minister of India resigned this morning" perfectly, and if that
article is from 2014 the entailment is a coincidence rather than a
confirmation. Nothing compared the two dates, because until now no article
carried a date at all — Google News ships RFC 2822 in <pubDate>, GNews and
NewsAPI ship ISO 8601 in publishedAt, and all of it was decoded and discarded.

Retrieval had the same blind spot from the other end. The feeds are recency-
RANKED but not recency-FILTERED, and NewsAPI was queried with
`sortBy=relevancy` — which for a breaking story returns last year's article
about the same subject, because it matches the words better.

Two failure directions are pinned here, and the second is the dangerous one:

  - An old article must not confirm a claim about today.
  - The guard must not fire on an undated claim, must not fire when the
    provider gave no date, and must never turn an old article into evidence
    AGAINST the claim. An old article is not proof that today's event did not
    happen.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import evidence_pipeline  # noqa: E402
import main  # noqa: E402
from claim_recency import (  # noqa: E402
    RECENT_WINDOW_DAYS,
    ClaimTime,
    document_is_stale,
    resolve_mode,
)
from providers import ProviderDiagnostic, SearchResult  # noqa: E402
from providers.dates import parse_iso8601, parse_rfc2822  # noqa: E402

NOW = datetime.now(timezone.utc)
LONG_AGO = datetime(2014, 5, 1, tzinfo=timezone.utc)
RECENT_CLAIM = "The prime minister of India resigned this morning"
UNDATED_CLAIM = "The prime minister of India resigned"


class TestDatesAreParsedFromEveryFormat(unittest.TestCase):
    """All three were already in the payloads and thrown away."""

    def test_rss_pubdate(self):
        parsed = parse_rfc2822("Tue, 02 Sep 2025 14:32:00 GMT")
        self.assertEqual(parsed, datetime(2025, 9, 2, 14, 32, tzinfo=timezone.utc))

    def test_iso_with_a_trailing_z(self):
        parsed = parse_iso8601("2025-09-02T14:32:00Z")
        self.assertEqual(parsed, datetime(2025, 9, 2, 14, 32, tzinfo=timezone.utc))

    def test_an_offset_is_converted_to_utc(self):
        parsed = parse_iso8601("2025-09-02T14:32:00+05:30")
        self.assertEqual(parsed, datetime(2025, 9, 2, 9, 2, tzinfo=timezone.utc))

    def test_everything_unparseable_becomes_none_rather_than_a_guess(self):
        """A wrong date would be used to discard real coverage."""
        for value in ("yesterday-ish", "", None, "2025-13-45T99:99:99Z"):
            with self.subTest(value=value):
                self.assertIsNone(parse_iso8601(value))
                self.assertIsNone(parse_rfc2822(value))


class TestModeResolution(unittest.TestCase):

    def test_the_claim_s_own_wording_selects_recent(self):
        for claim in ("The PM resigned this morning",
                      "Breaking: the central bank raised rates",
                      "The minister quit yesterday",
                      "Three people died earlier today"):
            with self.subTest(claim=claim):
                self.assertEqual(resolve_mode(claim).mode, "recent")

    def test_an_undated_claim_is_historical(self):
        for claim in ("The Eiffel Tower is in Paris",
                      "Nixon resigned in 1974",
                      UNDATED_CLAIM):
            with self.subTest(claim=claim):
                self.assertEqual(resolve_mode(claim).mode, "historical")

    def test_an_explicit_choice_beats_the_wording(self):
        """A story seen on television is typed without a time word at all."""
        self.assertEqual(resolve_mode(UNDATED_CLAIM, "recent").mode, "recent")
        self.assertEqual(resolve_mode(RECENT_CLAIM, "historical").mode, "historical")

    def test_an_unknown_mode_falls_back_to_auto(self):
        self.assertEqual(resolve_mode(RECENT_CLAIM, "sideways").mode, "recent")

    def test_the_window_is_only_set_in_recent_mode(self):
        self.assertEqual(resolve_mode(RECENT_CLAIM).window_days, RECENT_WINDOW_DAYS)
        self.assertIsNone(resolve_mode(UNDATED_CLAIM).window_days)

    def test_the_mode_explains_itself(self):
        self.assertIn("this morning", resolve_mode(RECENT_CLAIM).reason)
        self.assertIn("you selected", resolve_mode(UNDATED_CLAIM, "recent").reason)


class TestStalenessNeverFiresOnDoubt(unittest.TestCase):
    """Each of these would delete real evidence if it misfired."""

    RECENT = ClaimTime(anchor="recent", marker="this morning")
    HISTORICAL = ClaimTime(anchor="none")

    def test_an_old_document_is_stale_for_a_recent_claim(self):
        self.assertTrue(document_is_stale(self.RECENT, LONG_AGO, NOW))

    def test_a_document_inside_the_window_is_not(self):
        self.assertFalse(document_is_stale(self.RECENT, NOW - timedelta(days=3), NOW))

    def test_a_document_just_outside_the_search_window_is_not_yet_stale(self):
        """Retrieval asks for 30 days; evidence is not thrown away at 31."""
        self.assertFalse(document_is_stale(
            self.RECENT, NOW - timedelta(days=RECENT_WINDOW_DAYS + 5), NOW))

    def test_an_undated_claim_is_never_affected(self):
        self.assertFalse(document_is_stale(self.HISTORICAL, LONG_AGO, NOW))

    def test_a_document_with_no_date_is_never_stale(self):
        """Wikipedia and DuckDuckGo supply none; unknown is not old."""
        self.assertFalse(document_is_stale(self.RECENT, None, NOW))

    def test_a_future_timestamp_is_not_stale(self):
        self.assertFalse(document_is_stale(
            self.RECENT, NOW + timedelta(days=2), NOW))


class _NLI:
    is_available = True
    is_ready = True
    status = {"status": "ready", "enabled": True, "model": "stub", "error": None}

    def score_many(self, claim, passages):
        return [{
            "entailment": 0.92 if "resigned" in p.lower() else 0.05,
            "contradiction": 0.02, "neutral": 0.06, "available": True,
        } for p in passages]


class TestThroughTheApi(unittest.TestCase):

    BODY = ("India's prime minister resigned on Tuesday after coalition talks "
            "collapsed, his office confirmed.")

    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(main.app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def check(self, statement, mode="auto", published=None):
        results = [
            SearchResult(url=f"https://{d}/story", title="India PM resigns",
                         snippet=self.BODY, text=(self.BODY + " ") * 25,
                         provider="google_news", source=d, published=published)
            for d in ("reuters.com", "apnews.com", "bbc.co.uk")
        ]
        self.asked_for = None

        def fake_search(queries, **kwargs):
            self.asked_for = kwargs.get("recent_days")
            diagnostics = [ProviderDiagnostic(
                provider=name, query="q", enabled=True, status="success",
                raw_result_count=3, new_result_count=3)
                for name in ("google_news", "gnews")]
            return list(results), diagnostics

        nli = _NLI()
        with patch.object(evidence_pipeline, "search_all_providers", fake_search), \
             patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
             patch.object(main, "get_nli_service", lambda: nli):
            response = self.client.post(
                "/api/check", json={"statement": statement, "mode": mode})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_a_fresh_article_confirms_a_claim_about_today(self):
        body = self.check(RECENT_CLAIM, published=NOW - timedelta(days=1))
        self.assertEqual(body["verification"]["status"], "supported")

    def test_a_2014_article_does_not_confirm_a_claim_about_today(self):
        """The failure this whole file is about."""
        body = self.check(RECENT_CLAIM, published=LONG_AGO)
        self.assertNotEqual(body["verification"]["status"], "supported")

    def test_the_reason_is_shown_rather_than_left_unexplained(self):
        body = self.check(RECENT_CLAIM, published=LONG_AGO)
        notes = [e.get("stance_note") for e in body["top_evidence"]]
        self.assertTrue(any(note and "years ago" in note for note in notes), notes)

    def test_old_coverage_is_not_reported_as_nobody_having_reported_it(self):
        """"I found this, but from 2014" is not "this never happened"."""
        body = self.check(RECENT_CLAIM, published=LONG_AGO)
        self.assertNotEqual(
            body["verification"]["status"], "unsupported_no_coverage")

    def test_an_old_article_is_never_evidence_against_the_claim(self):
        body = self.check(RECENT_CLAIM, published=LONG_AGO)
        self.assertNotEqual(body["verification"]["status"], "contradicted")
        stances = {e["stance"] for e in body["top_evidence"]}
        self.assertNotIn("contradicts", stances)

    def test_the_same_old_article_still_confirms_an_undated_claim(self):
        """The control: age only matters when the claim is about now."""
        body = self.check(UNDATED_CLAIM, published=LONG_AGO)
        self.assertEqual(body["verification"]["status"], "supported")

    def test_recent_mode_asks_providers_for_a_date_window(self):
        self.check(RECENT_CLAIM, published=NOW)
        self.assertEqual(self.asked_for, RECENT_WINDOW_DAYS)

    def test_historical_mode_asks_for_no_date_window(self):
        self.check(UNDATED_CLAIM, published=NOW)
        self.assertIsNone(self.asked_for)

    def test_an_explicit_mode_reaches_retrieval(self):
        self.check(UNDATED_CLAIM, mode="recent", published=NOW)
        self.assertEqual(self.asked_for, RECENT_WINDOW_DAYS)

    def test_a_pasted_breaking_headline_gets_a_date_window(self):
        """End to end: the label reaches retrieval, not just resolve_mode."""
        self.check("BREAKING: The prime minister of India resigned",
                   published=NOW)
        self.assertEqual(self.asked_for, RECENT_WINDOW_DAYS)

    def test_the_response_says_which_mode_ran_and_why(self):
        coverage = self.check(RECENT_CLAIM, published=NOW)["coverage"]
        self.assertEqual(coverage["mode"], "recent")
        self.assertEqual(coverage["window_days"], RECENT_WINDOW_DAYS)
        self.assertIn("this morning", coverage["reason"])

    def test_the_publication_date_reaches_the_evidence_cards(self):
        body = self.check(RECENT_CLAIM, published=NOW - timedelta(days=1))
        self.assertTrue(all(e["published"] for e in body["top_evidence"]))

    def test_a_missing_date_is_carried_as_null_not_invented(self):
        body = self.check(UNDATED_CLAIM, published=None)
        self.assertTrue(all(e["published"] is None for e in body["top_evidence"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTheWireLabelSurvivesNormalisation(unittest.TestCase):
    """A bug created by combining two correct features.

    Normalisation strips the wire-service label off the front of a submission,
    which is right for search — "BREAKING:" is packaging, not proposition, and
    useless as a query term. But mode detection then ran on the normalised
    text, and that label is the strongest thing a pasted headline says about
    WHEN. Measured before the fix:

        "BREAKING: The United States banned Google"
          normalised -> "The United States banned Google"
          mode       -> historical

    So every pasted headline — the single most common way a breaking-news
    claim arrives — searched without a date window and accepted coverage of
    any age. Two separate faults: the labels that WERE recency markers were
    stripped before detection, and "JUST IN"/"URGENT" were never markers.
    """

    RECENCY_LABELS = [
        "BREAKING: The United States banned Google",
        "JUST IN: The prime minister of India resigned",
        "DEVELOPING: Three people died in the collapse",
        "URGENT: The central bank raised interest rates",
        "UPDATE: The minister has now resigned",
    ]

    # Labels that say what KIND of piece it is, not when it happened.
    NON_TIME_LABELS = [
        "EXCLUSIVE: an internal memo shows the CEO knew",
        "ANALYSIS: what the ruling means for the tech industry",
        "OPINION: the ban would be a mistake",
    ]

    def resolve(self, raw):
        from claim_normalizer import normalize_claim
        return resolve_mode(normalize_claim(raw), "auto", submitted=raw)

    def test_the_label_is_stripped_from_the_claim(self):
        """The premise: the signal really is gone from the normalised text."""
        from claim_normalizer import normalize_claim
        for raw in self.RECENCY_LABELS:
            with self.subTest(raw=raw):
                normalised = normalize_claim(raw)
                self.assertNotIn("BREAKING", normalised.upper()[:12])
                self.assertEqual(resolve_mode(normalised).mode, "historical")

    def test_a_pasted_breaking_headline_still_runs_in_recent_mode(self):
        for raw in self.RECENCY_LABELS:
            with self.subTest(raw=raw):
                self.assertEqual(self.resolve(raw).mode, "recent")

    def test_a_label_that_is_not_about_time_does_not_force_recent(self):
        for raw in self.NON_TIME_LABELS:
            with self.subTest(raw=raw):
                self.assertEqual(self.resolve(raw).mode, "historical")

    def test_the_explanation_says_where_the_signal_came_from(self):
        self.assertIn("BREAKING", self.resolve(self.RECENCY_LABELS[0]).reason)
        self.assertIn("this morning", self.resolve(RECENT_CLAIM).reason)

    def test_an_explicit_mode_still_overrides_the_label(self):
        from claim_normalizer import normalize_claim
        raw = self.RECENCY_LABELS[0]
        resolved = resolve_mode(normalize_claim(raw), "historical", submitted=raw)
        self.assertEqual(resolved.mode, "historical")


class TestTheModeIsAHintNotARequirement(unittest.TestCase):
    """An unrecognised mode must not fail the whole fact-check.

    `resolve_mode()` has always normalised anything it does not recognise to
    "auto", but a `pattern=` constraint on the request model made that fallback
    unreachable: an unknown value was rejected with a raw Pydantic 422 before
    any of the logic ran. It also disagreed with the Express proxy, which
    coerces unknown modes to "auto" — so the same request succeeded through the
    UI and failed against the API directly.

    This is a search hint with a safe default. Refusing to check a claim
    because the hint was malformed is the wrong trade.
    """

    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(main.app)
        cls.client = cls._cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)

    def check(self, mode):
        nli = _NLI()
        with patch.object(
            evidence_pipeline, "search_all_providers",
            lambda q, **k: ([], [ProviderDiagnostic(
                provider="google_news", query="q", enabled=True,
                status="no_results")])
        ), patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
                patch.object(main, "get_nli_service", lambda: nli):
            return self.client.post(
                "/api/check", json={"statement": RECENT_CLAIM, "mode": mode})

    def test_an_unknown_mode_is_normalised_rather_than_refused(self):
        for mode in ("sideways", "", "yesterday-ish"):
            with self.subTest(mode=mode):
                response = self.check(mode)
                self.assertEqual(response.status_code, 200, response.text)

    def test_an_unknown_mode_falls_back_to_reading_the_claim(self):
        """RECENT_CLAIM says "this morning", so auto resolves to recent."""
        body = self.check("sideways").json()
        self.assertEqual(body["coverage"]["mode"], "recent")

    def test_case_is_not_significant(self):
        self.assertEqual(self.check("RECENT").json()["coverage"]["mode"], "recent")
        self.assertEqual(
            self.check("Historical").json()["coverage"]["mode"], "historical")

    def test_a_valid_mode_still_wins_over_the_wording(self):
        body = self.check("historical").json()
        self.assertEqual(body["coverage"]["mode"], "historical")

    def test_the_request_model_carries_no_pattern_constraint(self):
        """The drift guard: re-adding one silently resurrects the 422."""
        field = main.CheckRequest.model_fields["mode"]
        patterns = [
            getattr(m, "pattern", None) for m in getattr(field, "metadata", [])
        ]
        self.assertTrue(
            all(p is None for p in patterns),
            "a pattern on `mode` makes resolve_mode()'s fallback unreachable",
        )
