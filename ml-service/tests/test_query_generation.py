"""Query generation — what the system actually asks the search providers.

WHY THIS EXISTS:
Only the first four generated queries are ever dispatched, so their ORDER is
the retrieval strategy. It used to be generation order, which put the four
least useful queries first. For "The prime minister of India resigned this
morning" the system sent:

    "The prime minister of India resigned this morning"   (exact phrase)
    The prime minister of India resigned this morning
    "India" "morning"
    India morning

An exact-phrase search for the user's own paraphrase matches nothing, and two
slots went to "morning" because that is the token the predicate regex returned
first. No dispatched query contained "resigned". Everything downstream —
relevance, NLI, the verdict — was working from documents retrieved by those
queries.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from event_vocabulary import events_in, surface_forms_in  # noqa: E402
from query_generator import QueryGenerator  # noqa: E402

# What the pipeline actually sends (evidence_pipeline slices to four).
DISPATCHED = 4


class QueryCaseMixin:
    @classmethod
    def setUpClass(cls):
        cls.generator = QueryGenerator()

    def dispatched(self, claim: str) -> list[str]:
        return [q["query"] for q in self.generator.generate_queries(claim)[:DISPATCHED]]

    def purposes(self, claim: str) -> list[str]:
        return [q["purpose"] for q in self.generator.generate_queries(claim)[:DISPATCHED]]


class TestTheActionReachesTheQueries(QueryCaseMixin, unittest.TestCase):
    """The claim's verb is the thing most likely to find its coverage."""

    def test_a_resignation_claim_searches_for_the_resignation(self):
        queries = self.dispatched("The prime minister of India resigned this morning")
        self.assertTrue(
            any("resign" in q.lower() for q in queries),
            f"no dispatched query mentions the event: {queries}",
        )

    def test_the_time_of_day_does_not_displace_the_verb(self):
        """"morning" was extracted as the predicate and used twice."""
        queries = self.dispatched("The prime minister of India resigned this morning")
        morning_queries = [q for q in queries if "morning" in q.lower() and "resign" not in q.lower()]
        self.assertLessEqual(len(morning_queries), 1)

    def test_a_ban_claim_searches_for_the_ban(self):
        queries = self.dispatched("The United States is Going to ban Google across all its cities")
        self.assertTrue(any("ban" in q.lower() for q in queries), queries)

    def test_a_purchase_claim_searches_for_the_purchase(self):
        queries = self.dispatched("Elon Musk bought the Eiffel Tower for 3 trillion dollars")
        self.assertTrue(any("bought" in q.lower() for q in queries), queries)

    def test_the_subject_and_the_action_appear_together(self):
        queries = self.dispatched("The prime minister of India resigned this morning")
        self.assertTrue(
            any("india" in q.lower() and "resign" in q.lower() for q in queries),
            queries,
        )


class TestQueryOrdering(QueryCaseMixin, unittest.TestCase):

    def test_the_plain_claim_is_the_first_query(self):
        self.assertEqual(self.purposes("The prime minister of India resigned this morning")[0],
                         "proposition")

    def test_the_exact_phrase_query_is_never_dispatched_when_better_ones_exist(self):
        """An exact-phrase search for a paraphrase returns nothing from a news index."""
        for claim in (
            "The prime minister of India resigned this morning",
            "The United States is Going to ban Google across all its cities",
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
        ):
            with self.subTest(claim=claim):
                self.assertNotIn("exact_claim", self.purposes(claim))

    def test_every_dispatched_query_is_distinct(self):
        queries = self.dispatched("The United States is Going to ban Google across all its cities")
        self.assertEqual(len(queries), len(set(q.lower() for q in queries)))

    def test_a_claim_with_no_recognised_event_still_produces_queries(self):
        queries = self.dispatched("The name of united states is being changed to india by 2050.")
        self.assertGreaterEqual(len(queries), 2)
        self.assertTrue(any("united states" in q.lower() for q in queries))


class TestRenameQueriesAreGated(QueryCaseMixin, unittest.TestCase):
    """Three rename queries used to fire on every two-entity claim."""

    def test_a_purchase_claim_does_not_search_for_a_name_change(self):
        all_queries = [
            q["query"].lower()
            for q in self.generator.generate_queries(
                "Elon Musk bought the Eiffel Tower for 3 trillion dollars"
            )
        ]
        self.assertFalse(
            any("renamed" in q or "name change" in q for q in all_queries),
            f"rename queries leaked into a purchase claim: {all_queries}",
        )

    def test_an_actual_rename_claim_still_gets_them(self):
        all_queries = [
            q["query"].lower()
            for q in self.generator.generate_queries(
                "The name of united states is being changed to india by 2050."
            )
        ]
        self.assertTrue(any("name change" in q or "renamed" in q for q in all_queries))


class TestEventVocabulary(unittest.TestCase):
    """The shared vocabulary both retrieval and relevance depend on."""

    def test_surface_forms_are_returned_in_the_claim_s_own_wording(self):
        self.assertEqual(surface_forms_in("The minister resigned yesterday")[0], "resigned")

    def test_multi_word_forms_win_over_their_substrings(self):
        forms = surface_forms_in("The minister steps down tomorrow")
        self.assertIn("steps down", forms)

    def test_an_unrecognised_verb_yields_nothing_rather_than_a_guess(self):
        self.assertEqual(events_in("The committee deliberated at length"), set())

    def test_different_wordings_map_to_the_same_event(self):
        self.assertEqual(events_in("he resigned"), events_in("he steps down"))

    def test_both_stages_read_the_same_vocabulary(self):
        """Retrieval and relevance must agree on what the claim is about."""
        from relevance_filter import RelevanceFilter
        claim = "The prime minister of India resigned this morning"
        self.assertEqual(RelevanceFilter()._claim_events(claim), events_in(claim))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestEntitylessClaims(QueryCaseMixin, unittest.TestCase):
    """Claims with no named subject at all.

    Almost every generator here is keyed on an entity, so a claim like "the
    vaccine is 95% effective" dispatched two queries: the sentence, and the
    same sentence in quotes. Meanwhile the fact-check generator built its
    query from whatever the predicate regex returned first, producing
    " 95 fact check verify" — a leading space, a bare number, no subject.
    """

    ENTITYLESS = [
        "The vaccine is 95% effective according to the trial",
        "Officials said the bridge collapsed overnight",
        "A leaked memo shows the CEO knew about the defect",
    ]

    def test_a_content_word_query_is_generated(self):
        for claim in self.ENTITYLESS:
            with self.subTest(claim=claim):
                self.assertGreaterEqual(len(self.dispatched(claim)), 2)
                self.assertIn("entity_action", self.purposes(claim))

    def test_no_query_is_malformed(self):
        for claim in self.ENTITYLESS:
            for query in self.dispatched(claim):
                with self.subTest(claim=claim, query=query):
                    self.assertEqual(query, query.strip(), "leading/trailing space")
                    self.assertGreater(len(query.split()), 1, "a one-token query")

    def test_no_fact_check_query_without_a_subject(self):
        """"' 95 fact check verify' finds nothing and costs a dispatched slot."""
        for claim in self.ENTITYLESS:
            with self.subTest(claim=claim):
                self.assertNotIn("fact_check", self.purposes(claim))

    def test_claims_with_an_entity_still_prefer_entity_queries(self):
        claim = "The prime minister of India resigned this morning"
        queries = self.dispatched(claim)
        self.assertTrue(any("India" in q for q in queries))


class TestCollectiveNounsAreNotEntities(unittest.TestCase):
    """English capitalises the first word of a sentence."""

    def entities(self, claim):
        from claim_decomposer import decompose_claim
        return decompose_claim(claim).primary_entities

    def test_a_sentence_initial_role_noun_is_not_an_entity(self):
        for claim, absent in (
            ("Scientists discovered a new species in the Amazon", "Scientists"),
            ("Officials said the bridge collapsed overnight", "Officials"),
            ("Reports suggest the merger will be blocked", "Reports"),
            ("Police arrested three people in London", "Police"),
        ):
            with self.subTest(claim=claim):
                self.assertNotIn(absent, self.entities(claim))

    def test_real_names_containing_one_are_kept_whole(self):
        self.assertIn("Metropolitan Police",
                      self.entities("Metropolitan Police arrested three people in London"))
        self.assertIn("Doctors Without Borders",
                      self.entities("Doctors Without Borders opened a clinic in Sudan"))

    def test_the_real_entity_in_the_claim_survives(self):
        self.assertIn("Amazon", self.entities("Scientists discovered a new species in the Amazon"))
