"""End-to-end: a viral false claim, with the evidence pool it really produces.

WHY THIS EXISTS:
Every other test isolates one stage. This one runs the whole pipeline against
the situation the system exists for, and it is the situation where the
individual bugs *compounded* — each one alone was survivable, and together
they guaranteed the wrong answer.

The scenario is a fabricated claim that has gone viral. The realistic evidence
pool is lopsided: many low-quality posts repeating the claim in its own
wording, and a couple of credible sources debunking it in theirs. Walking that
pool through the pipeline as it was:

  1. Search sent four queries, none containing the claim's verb, because the
     predicate regex picked a different token.
  2. Relevance ranked the rumour posts above the debunkings — they share the
     claim's exact wording, which is what lexical relevance rewards.
  3. Only the top eight were classified, so both credible sources were
     discarded unread.
  4. Had the fact-check been read, its quotation of the claim ("Posts claim
     …") scored as strong entailment, and both scores were taken from that one
     passage — so it would have counted as SUPPORTING the claim at 0.95
     source weight.
  5. Dedup could merge a debunking headline with the rumour it contradicts,
     since they differ by one word.
  6. Neutral coverage diluted the direction averages, so a real signal could
     be washed out by background articles.

Any one of those flips the verdict. This test asserts the composed outcome.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import evidence_pipeline  # noqa: E402
import main  # noqa: E402
from providers import ProviderDiagnostic, SearchResult  # noqa: E402


CLAIM = "The United States banned Google across all its cities"

# Eight low-quality posts repeating the claim, in the claim's own words.
RUMOUR_POSTS = [
    ("viralnews1.example", "Google ban rumours spread across all US cities",
     "Reports of a Google ban in all US cities spread widely this week."),
    ("viralnews2.example", "Google banned in all United States cities, users say",
     "A Google ban across all US cities was reported by many users."),
    ("viralnews3.example", "US cities Google ban: everything we know",
     "The Google ban across all US cities explained in detail."),
    ("viralnews4.example", "All US cities affected by Google ban claims",
     "The claimed Google ban across all cities is spreading fast."),
    ("viralnews5.example", "Google ban across every US city discussed",
     "Discussion of a Google ban across US cities continues online."),
    ("viralnews6.example", "US Google ban in all cities trends online",
     "The Google ban in all US cities trended for several hours."),
    ("viralnews7.example", "Google ban all US cities update",
     "Update on the Google ban across all US cities as it develops."),
    ("viralnews8.example", "Google banned across US cities, more claims emerge",
     "More reports that Google was banned across US cities have emerged."),
]

# The two sources that actually know something — written in their own words.
CREDIBLE_SOURCES = [
    ("politifact.com", "Fact check: the US has not banned Google",
     "Posts claim the United States banned Google in all its cities. "
     "This is false. No such prohibition exists and no bill has been introduced."),
    ("reuters.com", "No US prohibition on Google, regulators confirm",
     "Regulators confirmed that no nationwide prohibition on Google is in force."),
]


class ScenarioNLI:
    """An NLI double that behaves the way a real model does on this pool.

    The rumour posts restate the claim, so they entail it. The fact-check
    contains BOTH a strong entailment (its quotation of the claim) and a
    strong contradiction (its refutation) — which is the shape that inverted
    the verdict.
    """

    is_available = True
    status = {"status": "ready", "enabled": True, "model": "stub", "error": None}
    is_ready = True

    def score_many(self, claim, passages):
        scores = []
        for passage in passages:
            lowered = passage.lower()
            if "posts claim" in lowered:
                scores.append(self._s(0.88, 0.03))        # the quoted claim
            elif "this is false" in lowered or "no such prohibition" in lowered:
                scores.append(self._s(0.04, 0.86))        # the refutation
            elif "no nationwide prohibition" in lowered or "has not banned" in lowered:
                scores.append(self._s(0.05, 0.81))        # the debunking headline
            elif "ban" in lowered and "google" in lowered:
                scores.append(self._s(0.71, 0.04))        # a rumour restating it
            else:
                scores.append(self._s(0.05, 0.04))
        return scores

    @staticmethod
    def _s(entail, contradict):
        return {"entailment": entail, "contradiction": contradict,
                "neutral": max(0.0, 1 - entail - contradict), "available": True}


class TestViralFalseClaim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._client_cm = TestClient(main.app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def check(self, rows):
        results = [
            SearchResult(url=f"https://{domain}/story-{i}", title=title,
                         snippet=body, text=(body + " ") * 25,
                         provider="google_news", source=domain)
            for i, (domain, title, body) in enumerate(rows)
        ]
        diagnostics = [ProviderDiagnostic(
            provider="google_news", query="q", enabled=True, status="success",
            raw_result_count=len(results), new_result_count=len(results),
        )]
        nli = ScenarioNLI()
        with patch.object(evidence_pipeline, "search_all_providers",
                          lambda q, **k: (list(results), diagnostics)), \
             patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
             patch.object(main, "get_nli_service", lambda: nli):
            response = self.client.post("/api/check", json={"statement": CLAIM})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    # ── The composed outcome ─────────────────────────────────────────
    def test_the_claim_is_not_reported_as_supported(self):
        """The failure this whole scenario is about."""
        body = self.check(RUMOUR_POSTS + CREDIBLE_SOURCES)
        self.assertNotEqual(
            body["verification"]["status"], "supported",
            "eight posts repeating a false claim are not evidence that it is true",
        )

    def test_the_credible_sources_are_read_at_all(self):
        body = self.check(RUMOUR_POSTS + CREDIBLE_SOURCES)
        publishers = {e.get("publisher") or e.get("source") for e in body["top_evidence"]}
        self.assertTrue(
            {"politifact.com", "reuters.com"} & publishers,
            f"no credible source survived selection: {sorted(publishers)}",
        )

    def test_the_fact_check_is_not_counted_as_supporting(self):
        body = self.check(RUMOUR_POSTS + CREDIBLE_SOURCES)
        fact_check = next(
            (e for e in body["top_evidence"]
             if "politifact" in (e.get("publisher") or e.get("source", ""))),
            None,
        )
        self.assertIsNotNone(fact_check, "the fact-check must reach the evidence list")
        self.assertNotEqual(fact_check["stance"], "supports")

    def test_the_debunking_headline_is_not_deduped_against_the_rumour(self):
        """They differ by a negation, which is 0.9+ Jaccard on the tokens."""
        body = self.check(RUMOUR_POSTS + CREDIBLE_SOURCES)
        self.assertGreaterEqual(body["retrieval"]["candidate_count"], len(RUMOUR_POSTS) + 1)

    def test_contradicting_evidence_is_surfaced_to_the_user(self):
        body = self.check(RUMOUR_POSTS + CREDIBLE_SOURCES)
        self.assertGreater(
            body["evidence"]["contradicting_count"], 0,
            "the sources that refute the claim must appear as refuting it",
        )

    # ── Controls: the same machinery must not distort a true claim ───
    def test_a_genuinely_supported_claim_is_still_supported(self):
        """The guardrails must not make the system unable to say "yes"."""
        rows = [
            ("reuters.com", "India's prime minister resigns",
             "The prime minister resigned on Tuesday after coalition talks failed."),
            ("apnews.com", "Indian PM steps down",
             "The prime minister resigned, ending weeks of speculation."),
            ("bbc.com", "PM quits amid political turmoil",
             "The prime minister resigned after partners withdrew support."),
        ]
        results = [
            SearchResult(url=f"https://{d}/s{i}", title=t, snippet=b,
                         text=(b + " ") * 25, provider="google_news", source=d)
            for i, (d, t, b) in enumerate(rows)
        ]
        diagnostics = [ProviderDiagnostic(
            provider="google_news", query="q", enabled=True, status="success",
            raw_result_count=3, new_result_count=3,
        )]

        class SupportingNLI(ScenarioNLI):
            def score_many(self, claim, passages):
                return [
                    self._s(0.93, 0.02) if "resign" in p.lower() or "steps down" in p.lower()
                    else self._s(0.05, 0.04)
                    for p in passages
                ]

        nli = SupportingNLI()
        with patch.object(evidence_pipeline, "search_all_providers",
                          lambda q, **k: (list(results), diagnostics)), \
             patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
             patch.object(main, "get_nli_service", lambda: nli):
            body = self.client.post(
                "/api/check",
                json={"statement": "The prime minister of India resigned this morning"},
            ).json()

        self.assertEqual(body["verification"]["status"], "supported")
        self.assertGreaterEqual(body["evidence"]["independent_supporting"], 2)
        self.assertEqual(body["confidence"], "high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
