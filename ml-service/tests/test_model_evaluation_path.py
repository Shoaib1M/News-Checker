"""Evaluation must score the model on the inputs it is actually served.

WHY THIS EXISTS:
The shipped model is trained statement-only, with the five credit-history
columns forced to zero (`binary_truth_mlp.main()`). `evaluate_models.py` loaded
that model and then scored it on `build_text_input(test_df)` — the statement
PLUS subject, speaker, job, state, party and context — with real non-zero
history counts. It was measured on a distribution it had never seen, and the
number it produced went straight to the frontend's Model Evaluation page:

    reported on the site                      0.5691
    scored through the path a request takes   0.6188
    majority-class baseline                   0.5635

So the project under-reported its own model by five points, and the comparison
against Logistic Regression was drawn from a mismatch rather than from the
model. A second, subtler version of the same bug lived in
`evaluate_production_model.py`, whose comment claimed it "precisely mirrors
main.py" while transforming the raw statement — main.py goes through
`build_text_input()`, which prepends column-name tokens and creates boundary
bigrams. That was worth another 0.5 points (0.6235 claimed, 0.6188 real).

Both now call `make_prediction_features_batch()`, which `make_prediction_features()`
— the function `main.py` uses — also delegates to. These tests keep it that way.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import (  # noqa: E402
    HISTORY_COLUMNS,
    TEXT_FEATURE_COLUMNS,
    make_prediction_features,
    make_prediction_features_batch,
)


class _StubVectorizer:
    """Records what text it was asked to transform."""

    vocab_size = 4

    def __init__(self):
        self.seen: list[str] = []

    def transform(self, documents):
        documents = list(documents)
        self.seen.extend(documents)
        return np.ones((len(documents), self.vocab_size))


class TestOneFeaturePath(unittest.TestCase):

    def setUp(self):
        self.vectorizer = _StubVectorizer()
        self.train_max_values = np.ones((1, len(HISTORY_COLUMNS)))

    def test_single_and_batch_agree_exactly(self):
        single = make_prediction_features(
            self.vectorizer, self.train_max_values, "the minister resigned")
        batch = make_prediction_features_batch(
            _StubVectorizer(), self.train_max_values, ["the minister resigned"])
        np.testing.assert_allclose(single, batch)

    def test_the_single_row_form_delegates_rather_than_reimplementing(self):
        """Two implementations of the same construction is how this drifted."""
        source = (SERVICE_DIR / "binary_truth_mlp.py").read_text()
        body = source.split("def make_prediction_features(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("make_prediction_features_batch(", body)
        self.assertNotIn("np.hstack", body,
                         "make_prediction_features is rebuilding features itself")

    def test_metadata_defaults_to_what_the_api_sends(self):
        """Blank strings and zero counts — never a guessed speaker or history."""
        make_prediction_features_batch(
            self.vectorizer, self.train_max_values, ["the minister resigned"])
        (text,) = self.vectorizer.seen
        for column in TEXT_FEATURE_COLUMNS:
            self.assertIn(column, text)
        self.assertIn("the minister resigned", text)

    def test_history_columns_are_zero_by_default(self):
        features = make_prediction_features_batch(
            self.vectorizer, self.train_max_values, ["x"])
        history = features[:, -len(HISTORY_COLUMNS):]
        np.testing.assert_allclose(history, 0.0)

    def test_a_batch_matches_the_rows_built_one_at_a_time(self):
        statements = ["the minister resigned", "rates rose by 25 basis points", "x"]
        batch = make_prediction_features_batch(
            _StubVectorizer(), self.train_max_values, statements)
        rows = np.vstack([
            make_prediction_features(_StubVectorizer(), self.train_max_values, s)
            for s in statements
        ])
        np.testing.assert_allclose(batch, rows)


class TestTheEvaluatorsUseIt(unittest.TestCase):
    """Drift guards. If an evaluator starts building its own features again,
    its number stops describing the shipped model."""

    def source(self, name):
        """Code only. The first version of this test matched the explanatory
        comment that describes the bug, and passed or failed on prose."""
        lines = (SERVICE_DIR / name).read_text().splitlines()
        return "\n".join(
            line for line in lines if not line.lstrip().startswith("#")
        )

    def test_evaluate_models_scores_through_the_serving_path(self):
        source = self.source("evaluate_models.py")
        self.assertIn("make_prediction_features_batch(", source)

    def test_evaluate_models_no_longer_feeds_the_model_speaker_metadata(self):
        source = self.source("evaluate_models.py")
        self.assertNotIn("build_text_input(test_df)", source)
        self.assertNotIn("build_history_features(test_df", source)

    def test_the_production_evaluator_scores_through_the_serving_path(self):
        source = self.source("evaluate_production_model.py")
        self.assertIn("make_prediction_features_batch(", source)
        self.assertNotIn("vectorizer.transform(test_df", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
