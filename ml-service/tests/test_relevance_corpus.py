"""A labelled corpus for the relevance filter, so its quality can't silently drift.

WHY THIS EXISTS:
Every other test here pins one behaviour. This one measures the filter as a
whole against twenty hand-labelled (claim, document) pairs, because relevance
is a tradeoff and single assertions hide tradeoffs: it is easy to "fix" a
false positive by raising a threshold and quietly lose two real articles.

WHY RECALL IS WEIGHTED ABOVE PRECISION HERE:
A document this filter rejects is gone — nothing downstream can recover it,
and for a high-salience claim enough wrong rejections turn into "no credible
source reports this", which is a statement about the world. A document it
lets through still has to be classified by NLI before it counts as evidence,
and if NLI finds it says nothing, it is shown under "Related coverage" rather
than counted. The costs are not symmetric, so the thresholds should not be
tuned as if they were.

WHAT THE MEASUREMENT SAYS (2026-09-05):
    threshold 0.30-0.48   precision 0.91  recall 1.00  F1 0.95
    threshold 0.50+       precision 1.00  recall 0.80  F1 0.89

The current threshold sits mid-plateau. Raising it to buy that last point of
precision costs two genuinely relevant articles, which is the wrong trade for
the reason above.

The single remaining false positive is instructive: "Google blocked our
account and never told us why" scores *identically* to the true positive
"Court declines to outlaw Google search deals" — entity 0.50, action 1.0,
predicate 0.12, coherence 0.40, specificity 0.14. On this feature set they are
the same document. No threshold or weight separates them, and pretending
otherwise by tuning would only overfit. The difference is semantic, which is
exactly what NLI is for: it scores the personal complaint neutral, and the
frontend files it under "Related coverage" rather than counting it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from relevance_filter import RelevanceFilter  # noqa: E402


# (claim, title, body, should_be_included)
# Labels are what a careful reviewer would say: is this document about what
# the claim asserts, closely enough that showing it as evidence is defensible?
CORPUS: list[tuple[str, str, str, bool]] = [
    # ── Genuinely about the claim ────────────────────────────────────
    ("The prime minister of India resigned this morning",
     "India's prime minister resigns after coalition collapse",
     "The prime minister resigned on Tuesday, ending weeks of speculation.", True),
    ("The prime minister of India resigned this morning",
     "Indian PM steps down amid political turmoil",
     "He stepped down after coalition partners withdrew support.", True),
    ("The United States banned Google across all its cities",
     "US lawmakers propose nationwide ban on Google services",
     "A bill would ban Google across all US cities.", True),
    ("The United States banned Google across all its cities",
     "Court declines to outlaw Google search deals",
     "Judges rejected calls to prohibit the arrangements nationwide.", True),
    ("A four-day workweek improves productivity",
     "Trial finds four-day week improves productivity",
     "The pilot found a four-day workweek improves productivity.", True),
    ("A four-day workweek improves productivity",
     "Study casts doubt on four-day week gains",
     "Researchers said output fell under the shorter schedule.", True),
    ("Elon Musk bought Twitter for 44 billion dollars",
     "Musk completes $44bn Twitter takeover",
     "Elon Musk purchased Twitter in a deal valued at 44 billion dollars.", True),
    ("Apple announced a new manufacturing plant in India",
     "Apple unveils India manufacturing expansion",
     "Apple announced a new plant in India, its largest outside China.", True),
    ("Inflation in the United States rose last month",
     "US inflation rises again in monthly data",
     "Consumer prices in the United States rose 0.4% last month.", True),
    ("The World Health Organization declared the outbreak over",
     "WHO declares the outbreak officially over",
     "The World Health Organization announced the emergency has ended.", True),

    # ── Right subjects, wrong story ──────────────────────────────────
    ("The United States banned Google across all its cities",
     "Google expands advertising tools in the United States",
     "Google announced new ad products for US businesses.", False),
    ("The prime minister of India resigned this morning",
     "India cricket team wins series against Australia",
     "India beat Australia in the final match on Tuesday.", False),
    ("The United States banned Google across all its cities",
     "Google blocked our account and never told us why",
     "Google blocked us from the service without warning.", False),
    ("A four-day workweek improves productivity",
     "Local cafe opens a new branch downtown",
     "The cafe opened its second location this week.", False),
    ("Elon Musk bought Twitter for 44 billion dollars",
     "Tesla reports record quarterly deliveries",
     "The carmaker delivered more vehicles than analysts expected.", False),
    ("Apple announced a new manufacturing plant in India",
     "A job that changed me: working as a theatre usher",
     "Working as an usher changed my perspective on life.", False),
    ("Inflation in the United States rose last month",
     "Kennedy Center reportedly changed rules before vote",
     "The Kennedy Center made changes to voting procedures.", False),
    ("The World Health Organization declared the outbreak over",
     "Health ministry opens three new district clinics",
     "The clinics will serve rural communities from next month.", False),
    ("The prime minister of India resigned this morning",
     "Morning commuters face delays across the capital",
     "Traffic was heavy through the morning rush hour.", False),
    ("The United States banned Google across all its cities",
     "France moves to ban TikTok on government phones",
     "A French rule bans TikTok from official devices.", False),
]


def measure(filt: RelevanceFilter):
    """Return (precision, recall, false_positives, false_negatives)."""
    tp = fp = fn = 0
    false_positives, false_negatives = [], []
    for claim, title, body, expected in CORPUS:
        score = filt.assess_document_relevance(claim, title, body, body * 10)
        included = filt.should_include_document(score, strict=True)
        if included and expected:
            tp += 1
        elif included and not expected:
            fp += 1
            false_positives.append(title)
        elif not included and expected:
            fn += 1
            false_negatives.append(title)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, false_positives, false_negatives


class TestRelevanceCorpus(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.precision, cls.recall, cls.false_positives, cls.false_negatives = \
            measure(RelevanceFilter())

    def test_no_relevant_document_is_rejected(self):
        """Recall must stay perfect: a rejected document cannot be recovered."""
        self.assertEqual(
            self.recall, 1.0,
            f"relevant documents were filtered out: {self.false_negatives}",
        )

    def test_precision_does_not_regress(self):
        self.assertGreaterEqual(
            self.precision, 0.90,
            f"irrelevant documents passed the filter: {self.false_positives}",
        )

    def test_at_most_one_false_positive_survives(self):
        """Documented above: the survivor is inseparable on these features."""
        self.assertLessEqual(len(self.false_positives), 1)

    def test_demonyms_count_as_the_country(self):
        """"Indian PM steps down" scored ZERO entity match against India."""
        filt = RelevanceFilter()
        score = filt.assess_document_relevance(
            "The prime minister of India resigned this morning",
            "Indian PM steps down amid political turmoil",
            "He stepped down after coalition partners withdrew support.",
            "He stepped down after coalition partners withdrew support. " * 10,
        )
        self.assertEqual(score.entity_match_score, 1.0)

    def test_irregular_demonyms_are_handled(self):
        filt = RelevanceFilter()
        for claim_entity, headline_word in (
            ("China", "Chinese"), ("France", "French"), ("Japan", "Japanese"),
            ("Britain", "British"), ("Israel", "Israeli"),
        ):
            with self.subTest(entity=claim_entity):
                score = filt.assess_document_relevance(
                    f"{claim_entity} banned the service nationwide",
                    f"{headline_word} regulators banned the service",
                    f"{headline_word} regulators announced the ban this week.",
                    f"{headline_word} regulators announced the ban this week. " * 10,
                )
                self.assertEqual(score.entity_match_score, 1.0)

    def test_a_suffix_match_does_not_admit_an_unrelated_country(self):
        filt = RelevanceFilter()
        score = filt.assess_document_relevance(
            "India banned the service nationwide",
            "Indonesia banned the service nationwide",
            "Indonesian regulators announced the ban this week.",
            "Indonesian regulators announced the ban this week. " * 10,
        )
        self.assertEqual(score.entity_match_score, 0.0)


if __name__ == "__main__":
    precision, recall, fps, fns = measure(RelevanceFilter())
    print(f"precision {precision:.2f}  recall {recall:.2f}")
    for title in fns:
        print(f"  FALSE NEGATIVE  {title}")
    for title in fps:
        print(f"  FALSE POSITIVE  {title}")
    unittest.main(verbosity=2)
