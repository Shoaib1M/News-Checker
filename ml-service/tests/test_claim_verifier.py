import sys
from pathlib import Path
import unittest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from nli_service import NLIService, UnresolvableLabelError, _normalise_label
from claim_verifier import classify_source, extract_claims


class FakePipeline:
    def __call__(self, pairs, **_kwargs):
        rows = []
        for pair in pairs:
            if "fell" in pair["text"].lower():
                rows.append([
                    {"label": "LABEL_0", "score": 0.96},
                    {"label": "LABEL_1", "score": 0.03},
                    {"label": "LABEL_2", "score": 0.01},
                ])
            else:
                rows.append([
                    {"label": "LABEL_0", "score": 0.02},
                    {"label": "LABEL_1", "score": 0.03},
                    {"label": "LABEL_2", "score": 0.95},
                ])
        return rows


class NamedLabelPipeline:
    """Simulates a model whose config defines real label names — order in
    the output list must not matter, since matching is by substring."""
    def __call__(self, pairs, **_kwargs):
        return [[
            {"label": "neutral", "score": 0.02},
            {"label": "CONTRADICTION", "score": 0.90},
            {"label": "Entailment", "score": 0.08},
        ] for _ in pairs]


class ClaimVerifierTests(unittest.TestCase):
    def test_claim_extraction_keeps_distinct_declarative_claims(self):
        claims = extract_claims(
            "Inflation fell to 3 percent in July. Unemployment also declined last month."
        )
        self.assertEqual(len(claims), 2)
        self.assertIn("Inflation fell", claims[0])

    def test_source_classification_is_conservative(self):
        self.assertEqual(classify_source("https://www.cdc.gov/data").tier, "primary")
        self.assertEqual(classify_source("https://www.factcheck.org/story").tier, "fact-check")
        self.assertEqual(classify_source("https://example.net/story").weight, 0.0)

    def test_nli_uses_entailment_and_contradiction_labels(self):
        scorer = NLIService(
            model_name="cross-encoder/nli-deberta-v3-small",
            pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        )
        scores = scorer.score_many(
            "Inflation increased.",
            ["Inflation increased last month.", "Inflation fell last month."],
        )
        self.assertTrue(scores[0]["available"])
        self.assertGreater(scores[0]["entailment"], scores[0]["contradiction"])
        self.assertGreater(scores[1]["contradiction"], scores[1]["entailment"])

    def test_named_labels_are_order_independent(self):
        """A model with real label names must work regardless of model
        identity or output order — no lookup table needed for these."""
        scorer = NLIService(
            model_name="some-other-org/totally-unlisted-nli-model",
            pipeline_factory=lambda *_args, **_kwargs: NamedLabelPipeline(),
        )
        scores = scorer.score_many("A claim.", ["A passage."])
        self.assertTrue(scores[0]["available"])
        self.assertAlmostEqual(scores[0]["contradiction"], 0.90)
        self.assertAlmostEqual(scores[0]["entailment"], 0.08)
        self.assertAlmostEqual(scores[0]["neutral"], 0.02)

    def test_unrecognized_indexed_labels_abstain_instead_of_guessing(self):
        """An unlisted model emitting raw LABEL_N must fail safe, not guess
        a label order that could silently invert every verdict."""
        with self.assertRaises(UnresolvableLabelError):
            _normalise_label("LABEL_0", "some-other-org/totally-unlisted-nli-model")

        scorer = NLIService(
            model_name="some-other-org/totally-unlisted-nli-model",
            pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        )
        scores = scorer.score_many("A claim.", ["A passage that fell."])
        self.assertFalse(scores[0]["available"])
        self.assertEqual(scorer.status["status"], "failed")

    def test_model_failure_is_explicitly_unavailable(self):
        def failing_factory(*_args, **_kwargs):
            raise RuntimeError("offline")

        scores = NLIService(pipeline_factory=failing_factory).score_many(
            "A claim", ["A passage with enough words for the check."]
        )
        self.assertFalse(scores[0]["available"])
        self.assertEqual(scores[0]["entailment"], 0.0)


if __name__ == "__main__":
    unittest.main()
