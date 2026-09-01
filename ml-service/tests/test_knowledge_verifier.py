import sys
from pathlib import Path
import unittest

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from knowledge_verifier import assess_claim


class KnowledgeVerifierTests(unittest.TestCase):
    def test_foundational_true_claim(self):
        result = assess_claim("Water freezes at 0°C at sea level.")
        self.assertEqual(result["verdict"], "true")
        self.assertEqual(result["confidence"], "very high")

    def test_foundational_false_claim(self):
        self.assertEqual(assess_claim("A triangle has four sides.")["verdict"], "false")

    def test_subjective_claim_is_not_fact_checked(self):
        self.assertEqual(
            assess_claim("Pizza is the best food.")["status"],
            "not_objectively_verifiable",
        )

    def test_unknown_claim_uses_external_pipeline(self):
        self.assertIsNone(assess_claim("Company X announced Y yesterday."))


if __name__ == "__main__":
    unittest.main()
