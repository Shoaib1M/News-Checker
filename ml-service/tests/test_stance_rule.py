"""The stance rule, and the sweep that measures it.

WHY THIS EXISTS:
STANCE_THRESHOLD and STANCE_DOMINANCE decide, for every document the system
reads, whether it counts as supporting the claim, contradicting it, or neither.
`stance_sweep.py` measures them against a labelled corpus using the real NLI
model, which needs `transformers` and a model download and so cannot run in
this suite.

What CAN be pinned offline is everything the sweep's conclusion rests on: that
the rule it evaluates is the rule the pipeline runs, and that its arithmetic is
right. A sweep that measured a reimplementation would report on a decision
procedure the system does not use — which is exactly how three other rules in
this codebase drifted from their copies.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import stance_sweep  # noqa: E402
from evidence_pipeline import (  # noqa: E402
    STANCE_DOMINANCE,
    STANCE_THRESHOLD,
    decide_stance,
)


class TestTheRule(unittest.TestCase):

    def test_a_clear_entailment_supports(self):
        self.assertEqual(decide_stance(0.91, 0.02), "supports")

    def test_a_clear_contradiction_contradicts(self):
        self.assertEqual(decide_stance(0.03, 0.88), "contradicts")

    def test_neither_direction_reaching_the_threshold_is_unclear(self):
        self.assertEqual(decide_stance(0.30, 0.28), "unclear")

    def test_a_document_arguing_both_ways_without_a_margin_is_unclear(self):
        """0.80 against 0.75 is a dispute, not a position."""
        self.assertEqual(decide_stance(0.80, 0.75), "unclear")

    def test_a_dominant_direction_wins_even_when_both_fire(self):
        self.assertEqual(decide_stance(0.88, 0.40), "supports")
        self.assertEqual(decide_stance(0.40, 0.88), "contradicts")

    def test_the_threshold_is_exclusive(self):
        """Exactly at the threshold is not yet taking a side."""
        self.assertEqual(decide_stance(STANCE_THRESHOLD, 0.0), "unclear")

    def test_the_thresholds_are_overridable(self):
        """What makes a sweep possible at all."""
        self.assertEqual(decide_stance(0.5, 0.0, threshold=0.6), "unclear")
        self.assertEqual(decide_stance(0.5, 0.0, threshold=0.4), "supports")


class TestThePipelineUsesThisRule(unittest.TestCase):
    """The drift guard. If the pipeline stops calling it, this fails."""

    def test_the_pipeline_calls_decide_stance(self):
        source = (SERVICE_DIR / "evidence_pipeline.py").read_text()
        self.assertIn("decide_stance(support_score, contradiction_score)", source)

    def test_the_pipeline_holds_no_second_copy_of_the_comparison(self):
        source = (SERVICE_DIR / "evidence_pipeline.py").read_text()
        body = source.split("def decide_stance", 1)[1].split("\ndef ", 1)[0]
        rest = source.replace(body, "")
        self.assertNotIn("contradiction_score * STANCE_DOMINANCE", rest)

    def test_the_sweep_measures_the_pipeline_s_rule(self):
        self.assertIs(stance_sweep.decide_stance, decide_stance)


class TestSweepArithmetic(unittest.TestCase):
    """Scores stubbed in, so the metric computation itself is checked."""

    # (claim, passage, truth, entailment, contradiction)
    SCORED = [
        ("c", "p", "supports", 0.90, 0.02),     # correct support
        ("c", "p", "supports", 0.10, 0.05),     # missed support
        ("c", "p", "contradicts", 0.03, 0.85),  # correct contradiction
        ("c", "p", "neutral", 0.88, 0.01),      # invented a support
        ("c", "p", "neutral", 0.05, 0.04),      # correct neutral
    ]

    def test_accuracy_counts_every_row(self):
        report = stance_sweep.evaluate(self.SCORED, 0.35, 1.6)
        self.assertAlmostEqual(report["accuracy"], 3 / 5)

    def test_a_neutral_row_the_rule_calls_unclear_counts_as_correct(self):
        """The corpus and the rule name this class differently. Scoring them
        as different classes marked every correct neutral row as an error."""
        report = stance_sweep.evaluate(
            [("c", "p", "neutral", 0.05, 0.04)], 0.35, 1.6)
        self.assertAlmostEqual(report["accuracy"], 1.0)

    def test_an_invented_position_is_a_false_positive_not_a_miss(self):
        report = stance_sweep.evaluate(self.SCORED, 0.35, 1.6)
        self.assertEqual(report["supports"]["invented"], 1)
        self.assertEqual(report["supports"]["missed"], 1)
        self.assertAlmostEqual(report["supports"]["precision"], 1 / 2)
        self.assertAlmostEqual(report["supports"]["recall"], 1 / 2)

    def test_a_direction_with_no_errors_scores_perfectly(self):
        report = stance_sweep.evaluate(self.SCORED, 0.35, 1.6)
        self.assertAlmostEqual(report["contradicts"]["precision"], 1.0)
        self.assertAlmostEqual(report["contradicts"]["recall"], 1.0)

    def test_raising_the_threshold_trades_recall_for_precision(self):
        strict = stance_sweep.evaluate(self.SCORED, 0.95, 1.6)
        self.assertEqual(strict["supports"]["invented"], 0,
                         "a high threshold should stop inventing positions")
        self.assertEqual(strict["supports"]["recall"], 0.0)


class TestTheCorpus(unittest.TestCase):

    def test_every_row_is_labelled_with_a_stance_the_rule_can_return(self):
        for claim, passage, label in stance_sweep.CORPUS:
            with self.subTest(claim=claim):
                self.assertIn(label, {"supports", "contradicts", "neutral"})
                self.assertTrue(claim.strip() and passage.strip())

    def test_neutral_pairs_are_the_largest_group(self):
        """They are most of what retrieval returns, so they must dominate the
        corpus too — a corpus of clean entailment pairs would tune the
        thresholds for a distribution the system never sees."""
        labels = [label for _c, _p, label in stance_sweep.CORPUS]
        self.assertGreaterEqual(labels.count("neutral"), labels.count("supports"))

    def test_the_current_setting_is_on_the_grid(self):
        """Otherwise the sweep cannot report on what is actually shipped."""
        self.assertIn(STANCE_THRESHOLD, stance_sweep.GRID_THRESHOLDS)
        self.assertIn(STANCE_DOMINANCE, stance_sweep.GRID_DOMINANCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
