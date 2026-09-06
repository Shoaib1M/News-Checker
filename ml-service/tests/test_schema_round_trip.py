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

from pydantic import BaseModel  # noqa: E402

import main  # noqa: E402


def camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(word.capitalize() for word in rest)


def _nested_blocks() -> dict:
    """Every nested model on CheckResponse, discovered rather than listed.

    This was a hand-written dict, and it drifted the moment a block was added:
    `coverage` was introduced on CheckResponse, was not added here, and the
    guard therefore reported that every field round-tripped while the whole
    block was being dropped between the live check and the saved one. A
    drift test maintained by hand has the same failure mode as the
    duplicated rule it exists to catch.
    """
    blocks = {}
    for name, field in main.CheckResponse.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            blocks[name] = annotation
    return blocks


RESPONSE_BLOCKS = _nested_blocks()


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



class TestNonNumericStatusesAreInSync(unittest.TestCase):
    """The backend and the frontend must agree on which outcomes show a number.

    `combined_score` carries a placeholder 50 for outcomes that aren't a
    measurement of evidence. If the frontend doesn't know a status is one of
    those, it draws that 50 as a confident amber "middling" score — which is
    how a saved check of "asdkjh asdkjh" came to appear in the history list as
    a half-true claim.

    The rule lived in three places at once (the gauge, the history list, and
    ml-service). It is now one module per side, and this asserts the two sides
    list the same statuses.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = (REPO_ROOT / "client" / "src" / "verdictStates.js").read_text()

    def test_the_frontend_lists_every_backend_non_numeric_status(self):
        missing = [
            status for status in sorted(main.NON_NUMERIC_STATUSES)
            if not re.search(rf"^\s*{status}\s*:", self.source, re.MULTILINE)
        ]
        self.assertEqual(
            missing, [],
            "client/src/verdictStates.js is missing these, so the UI will draw "
            "a placeholder 50 for them as if it were a real score",
        )

    def test_the_frontend_lists_nothing_the_backend_scores_numerically(self):
        declared = set(re.findall(r"^\s*([a-z_]+)\s*:\s*\{", self.source, re.MULTILINE))
        extra = sorted(declared - main.NON_NUMERIC_STATUSES)
        self.assertEqual(
            extra, [],
            "these are scored numerically by the backend but hidden by the UI",
        )

    def test_every_non_numeric_status_has_a_label_and_colour(self):
        for status in sorted(main.NON_NUMERIC_STATUSES):
            with self.subTest(status=status):
                block = re.search(
                    rf"^\s*{status}\s*:\s*\{{(.*?)\}}", self.source,
                    re.MULTILINE | re.DOTALL,
                )
                self.assertIsNotNone(block)
                self.assertIn("label:", block.group(1))
                self.assertIn("color:", block.group(1))

    def test_both_score_gauge_and_history_panel_use_the_shared_module(self):
        """Either one keeping a private copy is how they drifted the first time."""
        for component in ("ScoreGauge.jsx", "HistoryPanel.jsx"):
            with self.subTest(component=component):
                source = (REPO_ROOT / "client" / "src" / "components" / component).read_text()
                self.assertIn("verdictStates", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
