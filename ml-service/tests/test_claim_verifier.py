import sys
from pathlib import Path
import unittest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from nli_service import NLIService
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
        scorer = NLIService(pipeline_factory=lambda *_args, **_kwargs: FakePipeline())
        scores = scorer.score_many(
            "Inflation increased.",
            ["Inflation increased last month.", "Inflation fell last month."],
        )
        self.assertTrue(scores[0]["available"])
        self.assertGreater(scores[0]["entailment"], scores[0]["contradiction"])
        self.assertGreater(scores[1]["contradiction"], scores[1]["entailment"])

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
