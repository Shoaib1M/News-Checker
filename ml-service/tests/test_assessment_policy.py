import sys
from pathlib import Path
import unittest

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from main import evidence_verdict_score, merge_claim_summaries


class AssessmentPolicyTests(unittest.TestCase):
    def test_no_evidence_is_never_rendered_as_false_or_true(self):
        summary = merge_claim_summaries([{
            "support": 0.0,
            "contradiction": 0.0,
            "net": 0.0,
            "status": "insufficient_evidence",
            "nli_available": False,
            "evidence_count": 0,
        }])
        self.assertEqual(summary["status"], "insufficient_evidence")
        self.assertEqual(evidence_verdict_score(summary), 50)

    def test_conflicting_atomic_claims_become_mixed(self):
        summary = merge_claim_summaries([
            {
                "support": 0.8, "contradiction": 0.1, "net": 0.7,
                "status": "supported", "nli_available": True, "evidence_count": 2,
            },
            {
                "support": 0.1, "contradiction": 0.8, "net": -0.7,
                "status": "contradicted", "nli_available": True, "evidence_count": 2,
            },
        ])
        self.assertEqual(summary["status"], "mixed")
        self.assertEqual(evidence_verdict_score(summary), 50)


if __name__ == "__main__":
    unittest.main()
