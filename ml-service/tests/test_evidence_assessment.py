import sys
from pathlib import Path
import unittest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import evidence_scraper
from claim_verifier import NLIScorer
from evidence_scraper import EvidenceResult, stance_summary


class FixedPipeline:
    def __call__(self, pairs, **_kwargs):
        return [[
            {"label": "contradiction", "score": 0.02},
            {"label": "neutral", "score": 0.03},
            {"label": "entailment", "score": 0.95},
        ] for _ in pairs]


class EvidenceAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.previous_scorer = evidence_scraper._nli_scorer
        evidence_scraper._nli_scorer = NLIScorer(
            pipeline_factory=lambda *_args, **_kwargs: FixedPipeline()
        )

    def tearDown(self):
        evidence_scraper._nli_scorer = self.previous_scorer

    def test_nli_evidence_is_ranked_above_unclassified_relevance(self):
        documents = [
            {
                "url": "https://www.reuters.com/report",
                "title": "Reuters report confirms vaccination coverage increased",
                "snippet": "",
                "text": "Vaccination coverage increased to 95 percent according to the CDC report.",
                "source": "Reuters",
                "provider": "test",
            },
            {
                "url": "https://example.net/repost",
                "title": "Vaccination coverage increased",
                "snippet": "",
                "text": "Vaccination coverage increased to 95 percent in this repost.",
                "source": "Example",
                "provider": "test",
            },
        ]
        results = evidence_scraper.rank_by_similarity(
            "Vaccination coverage increased to 95 percent.", documents
        )
        self.assertEqual(results[0].source_tier, "reporting")
        summary = evidence_scraper.stance_summary(results)
        self.assertEqual(summary["status"], "insufficient_evidence")

    def test_unavailable_nli_cannot_create_a_verdict(self):
        evidence_scraper._nli_scorer = NLIScorer(
            pipeline_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
        )
        results = evidence_scraper.rank_by_similarity(
            "Vaccination coverage increased.",
            [{
                "url": "https://www.cdc.gov/report",
                "title": "Vaccination coverage increased",
                "snippet": "",
                "text": "Vaccination coverage increased in the latest report.",
                "source": "CDC",
                "provider": "test",
            }],
        )
        self.assertEqual(evidence_scraper.stance_summary(results)["status"], "insufficient_evidence")

    def test_primary_source_has_more_weight_than_repeated_reporting(self):
        def result(source_tier, source_weight, support, contradiction, similarity):
            return EvidenceResult(
                url=f"https://{source_tier}.example/story",
                title="Claim report",
                snippet="",
                similarity=similarity,
                text_length=100,
                provider="test",
                source=source_tier,
                support_score=support,
                contradiction_score=contradiction,
                stance="supports" if support > contradiction else "contradicts",
                best_sentence="Claim report.",
                source_tier=source_tier,
                source_weight=source_weight,
                nli_available=True,
            )

        results = [
            result("primary", 1.0, 0.90, 0.05, 0.8),
            result("reporting", 0.8, 0.05, 0.90, 0.8),
            result("reporting", 0.8, 0.05, 0.90, 0.8),
        ]

        summary = stance_summary(results)

        self.assertEqual(summary["status"], "supported")
        self.assertGreater(summary["net"], 0)


if __name__ == "__main__":
    unittest.main()
