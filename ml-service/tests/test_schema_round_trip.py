"""Every API field survives the round trip to history and back.

WHY THIS EXISTS:
A check's result crosses three codebases: ml-service produces it, the Express
proxy persists it to MongoDB under camelCase names, the Mongoose schema has to
declare it, and the React client maps it back to snake_case when replaying
from history. A field added to the response and forgotten in any one of those
places disappears silently — replaying a saved check just renders less than
the live one did, with nothing failing anywhere.

That happened repeatedly while this branch was adding response fields
(`claim_kind`, `salience`, `independent_supporting`, `independent_contradicting`),
which is why it is now checked mechanically rather than by remembering.

This reads the other services' source as text. That is deliberately crude, but
the alternative is a shared schema definition, and the thing being guarded
against is precisely the kind of drift a human review misses.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVICE_DIR.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import main  # noqa: E402


def camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(word.capitalize() for word in rest)


# The nested blocks of CheckResponse, and where each is persisted.
RESPONSE_BLOCKS = {
    "verification": main.VerificationInfo,
    "ml": main.MLInfo,
    "retrieval": main.RetrievalInfo,
    "nli": main.NLIInfo,
    "evidence": main.EvidenceSummary,
}


class TestResponseFieldsSurviveHistory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.check_route = (REPO_ROOT / "server" / "routes" / "check.js").read_text()
        cls.check_model = (REPO_ROOT / "server" / "models" / "Check.js").read_text()
        cls.client_app = (REPO_ROOT / "client" / "src" / "App.jsx").read_text()

    def _fields(self):
        for block, model in RESPONSE_BLOCKS.items():
            for field in sorted(model.model_fields):
                yield block, field

    def test_every_field_is_written_to_the_database(self):
        missing = [
            f"{block}.{field}"
            for block, field in self._fields()
            if not re.search(rf"\b{camel(field)}\s*:", self.check_route)
        ]
        self.assertEqual(
            missing, [],
            "server/routes/check.js does not persist these response fields, so a "
            "saved check silently loses them",
        )

    def test_every_field_is_declared_in_the_mongoose_schema(self):
        missing = [
            f"{block}.{field}"
            for block, field in self._fields()
            if not re.search(rf"\b{camel(field)}\s*:", self.check_model)
        ]
        self.assertEqual(
            missing, [],
            "server/models/Check.js has no path for these fields, so Mongoose "
            "drops them on write regardless of what the route passes",
        )

    def test_every_field_is_restored_when_replaying_from_history(self):
        missing = []
        for block, field in self._fields():
            restored = (
                re.search(rf"\b{field}\s*:\s*full\.", self.client_app)
                or re.search(rf"\.{camel(field)}\b", self.client_app)
            )
            if not restored:
                missing.append(f"{block}.{field}")
        self.assertEqual(
            missing, [],
            "client/src/App.jsx does not map these back, so replaying a saved "
            "check renders less than the live check did",
        )

    def test_the_evidence_item_shape_is_persisted_too(self):
        """top_evidence is a sub-document with its own field list."""
        for field in sorted(main.EvidenceItem.model_fields):
            with self.subTest(field=field):
                self.assertRegex(
                    self.check_model, rf"\b{field}\s*:",
                    f"evidenceItemSchema has no path for {field}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
