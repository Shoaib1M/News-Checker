"""A document that states a different figure must not confirm the claim's.

WHY THIS EXISTS:
"The vaccine is 95% effective" and "the vaccine is 62% effective" differ by two
digits. Entities match, action matches, phrasing matches — every relevance
signal in the pipeline fires, and the passage is exactly the kind of lexical
neighbour a textual-entailment model scores as entailment. Nothing downstream
compared the numbers, so the article was recorded as SUPPORTING a figure it
contradicts, at whatever weight its publisher carries.

These tests pin both directions: the conflict is caught, and the guard stays
off claims where the number is incidental to what is being asserted.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from numeric_consistency import (  # noqa: E402
    conflicting_quantity,
    quantities_in,
)


class TestQuantityParsing(unittest.TestCase):

    def test_a_percentage_is_read_with_what_it_measures(self):
        (quantity,) = quantities_in("The vaccine is 95% effective")
        self.assertEqual(quantity.value, 95.0)
        self.assertEqual(quantity.kind, "percent")
        self.assertIn("effective", quantity.attribute)
        self.assertIn("vaccine", quantity.attribute)

    def test_the_number_itself_is_not_part_of_its_own_attribute(self):
        """Otherwise two unrelated figures written alike appear comparable."""
        (quantity,) = quantities_in("The vaccine is 95% effective")
        self.assertFalse(any(any(c.isdigit() for c in w) for w in quantity.attribute))

    def test_magnitude_words_scale_the_value(self):
        (quantity,) = quantities_in("a 3 trillion dollar purchase")
        self.assertEqual(quantity.value, 3e12)
        self.assertEqual(quantity.kind, "currency")

    def test_a_currency_symbol_is_enough_to_mark_money(self):
        (quantity,) = quantities_in("fined $500 million by regulators")
        self.assertEqual(quantity.kind, "currency")
        self.assertEqual(quantity.value, 5e8)

    def test_thousands_separators_are_not_read_as_separate_numbers(self):
        (quantity,) = quantities_in("5,000 people died")
        self.assertEqual(quantity.value, 5000.0)

    def test_percent_written_as_a_word(self):
        (quantity,) = quantities_in("unemployment rose to 4.2 percent")
        self.assertEqual(quantity.kind, "percent")
        self.assertEqual(quantity.value, 4.2)

    def test_a_bare_metre_measurement_is_not_read_as_millions(self):
        """"3 m" is three metres far more often than three million."""
        (quantity,) = quantities_in("a 3 m barrier")
        self.assertEqual(quantity.value, 3.0)


class TestConflictsAreCaught(unittest.TestCase):

    def test_a_different_percentage_for_the_same_attribute_conflicts(self):
        conflict = conflicting_quantity(
            "The vaccine is 95% effective",
            ["The vaccine is 62% effective against severe disease."],
        )
        self.assertIsNotNone(conflict)
        claimed, stated = conflict
        self.assertEqual(claimed.value, 95.0)
        self.assertEqual(stated.value, 62.0)

    def test_a_different_death_toll_conflicts(self):
        conflict = conflicting_quantity(
            "5,000 people died in the earthquake",
            ["Officials said 3,000 people died in the earthquake."],
        )
        self.assertIsNotNone(conflict)

    def test_a_different_fine_conflicts(self):
        conflict = conflicting_quantity(
            "Google was fined $5 billion by the European Commission",
            ["The Commission fined Google $2.4 billion over search results."],
        )
        self.assertIsNotNone(conflict)


class TestTheGuardStaysOffEverythingElse(unittest.TestCase):
    """Each of these would silently delete a good source if it misfired."""

    def test_a_document_stating_the_claim_s_figure_is_untouched(self):
        self.assertIsNone(conflicting_quantity(
            "The vaccine is 95% effective",
            ["Regulators disputed the trial design.",
             "The trial found the vaccine was 95% effective."],
        ))

    def test_the_figure_may_appear_in_any_passage_not_just_the_first(self):
        self.assertIsNone(conflicting_quantity(
            "The vaccine is 95% effective",
            ["Coverage reached 62% of adults by March.",
             "Efficacy was 95% in the vaccine arm."],
        ))

    def test_rounding_is_not_a_conflict(self):
        self.assertIsNone(conflicting_quantity(
            "The vaccine is 95% effective",
            ["The vaccine was 94.8% effective in the trial."],
        ))

    def test_a_figure_measuring_something_else_is_not_a_conflict(self):
        """The price is incidental; an article confirming the purchase counts."""
        self.assertIsNone(conflicting_quantity(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            ["Musk, whose net worth reached 400 billion dollars, "
             "confirmed the purchase on Tuesday."],
        ))

    def test_silence_about_the_figure_is_not_a_conflict(self):
        self.assertIsNone(conflicting_quantity(
            "The vaccine is 95% effective",
            ["The vaccine was approved by the regulator on Tuesday."],
        ))

    def test_a_claim_with_no_figure_is_never_affected(self):
        self.assertIsNone(conflicting_quantity(
            "The prime minister of India resigned",
            ["The prime minister resigned, one of 12 ministers to go this year."],
        ))

    def test_years_do_not_register_as_conflicting(self):
        """A date needs comparing against a timeline, not string matching."""
        self.assertIsNone(conflicting_quantity(
            "The United States is being renamed to India by 2050",
            ["A proposal would rename the country by 2035."],
        ))

    def test_a_percentage_is_never_compared_against_money(self):
        self.assertIsNone(conflicting_quantity(
            "Inflation reached 9% last year",
            ["Inflation cost households $9 billion last year."],
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── The same thing, through the real pipeline ────────────────────────

from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import evidence_pipeline  # noqa: E402
import main  # noqa: E402
from providers import ProviderDiagnostic, SearchResult  # noqa: E402


CLAIM = "The new vaccine is 95% effective at preventing infection"


class EntailingNLI:
    """Scores lexical near-neighbours as entailment, which is the problem.

    A sentence differing from the claim by two digits is, to a textual
    entailment model, the claim restated. This double does not exaggerate: it
    reproduces the behaviour the guard exists to survive, without depending on
    a particular checkpoint being installed.
    """

    is_available = True
    is_ready = True
    status = {"status": "ready", "enabled": True, "model": "stub", "error": None}

    def score_many(self, claim, passages):
        scores = []
        for passage in passages:
            lowered = passage.lower()
            if "vaccine" in lowered and "effective" in lowered:
                scores.append(self._s(0.91, 0.02))
            else:
                scores.append(self._s(0.05, 0.04))
        return scores

    @staticmethod
    def _s(entail, contradict):
        return {"entailment": entail, "contradiction": contradict,
                "neutral": max(0.0, 1 - entail - contradict), "available": True}


def _sources(efficacy: str):
    """Three independent publishers, all reporting the same efficacy figure."""
    body = (f"The trial found the new vaccine was {efficacy} effective at "
            f"preventing infection, researchers reported.")
    return [
        ("reuters.com", f"Vaccine {efficacy} effective in trial", body),
        ("apnews.com", f"Trial puts vaccine efficacy at {efficacy}", body),
        ("bbc.co.uk", f"New vaccine {efficacy} effective, researchers say", body),
    ]


class TestNumericConflictThroughThePipeline(unittest.TestCase):

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
        nli = EntailingNLI()
        with patch.object(evidence_pipeline, "search_all_providers",
                          lambda q, **k: (list(results), diagnostics)), \
             patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
             patch.object(main, "get_nli_service", lambda: nli):
            response = self.client.post("/api/check", json={"statement": CLAIM})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_sources_reporting_a_different_figure_do_not_confirm_the_claim(self):
        """Three real publishers, all saying 62%. The claim says 95%."""
        body = self.check(_sources("62%"))
        self.assertNotEqual(
            body["verification"]["status"], "supported",
            "articles reporting 62% cannot establish a 95% claim",
        )

    def test_no_such_source_is_counted_as_supporting(self):
        body = self.check(_sources("62%"))
        supporting = [e for e in body["top_evidence"] if e["stance"] == "supports"]
        self.assertEqual(supporting, [], "a 62% article was counted as support")

    def test_the_reason_is_stated_rather_than_left_as_an_unexplained_neutral(self):
        body = self.check(_sources("62%"))
        notes = [e.get("stance_note") for e in body["top_evidence"]]
        self.assertTrue(any(note and "62%" in note and "95%" in note for note in notes),
                        f"no evidence card explains the mismatch: {notes}")

    def test_the_control_still_works(self):
        """The guard must not make the system unable to confirm a figure."""
        body = self.check(_sources("95%"))
        self.assertEqual(body["verification"]["status"], "supported")
        supporting = [e for e in body["top_evidence"] if e["stance"] == "supports"]
        self.assertGreaterEqual(len(supporting), 2)
