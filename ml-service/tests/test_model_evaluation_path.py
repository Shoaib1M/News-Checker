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

    def test_a_statement_only_request_is_vectorised_as_the_bare_statement(self):
        """THE TRAIN/SERVE SKEW THIS PINS:

        `build_text_input` used to emit every column unconditionally, so a
        live request — which fills all the metadata with blanks — reached the
        vectorizer as

            "statement the minister resigned subject  speaker  job  state  party  context "

        while `binary_truth_mlp.main()` trained on the bare statement. Seven
        constant tokens and seven boundary bigrams, on every request, against
        a vocabulary that had never seen them. Measured, it cost 0.47 points
        (61.88% served vs 62.35% on the text it was trained on).

        An earlier version of this test asserted the opposite — that every
        column name appears in the text — and so pinned the defect in place.
        """
        make_prediction_features_batch(
            self.vectorizer, self.train_max_values, ["the minister resigned"])
        (text,) = self.vectorizer.seen
        self.assertEqual(text, "the minister resigned")

    def test_metadata_is_tagged_with_its_column_when_it_is_actually_present(self):
        """Skipping blanks must not mean losing metadata that was supplied:
        "texas" as a state and "texas" inside a claim are different features."""
        make_prediction_features_batch(
            self.vectorizer, self.train_max_values, ["the minister resigned"],
            state="Texas", party="republican")
        (text,) = self.vectorizer.seen
        self.assertEqual(text, "the minister resigned state Texas party republican")
        for column in TEXT_FEATURE_COLUMNS:
            if column not in ("statement", "state", "party"):
                self.assertNotIn(column, text)

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
        self.assertTrue("predict_proba_texts(" in source
                        or "make_prediction_features_batch(" in source)

    def test_evaluate_models_no_longer_feeds_the_model_speaker_metadata(self):
        source = self.source("evaluate_models.py")
        self.assertNotIn("build_text_input(test_df)", source)
        self.assertNotIn("build_history_features(test_df", source)

    def test_the_production_evaluator_scores_through_the_serving_path(self):
        source = self.source("evaluate_production_model.py")
        # predict_proba_texts() is make_prediction_features_batch() in chunks —
        # the dense serving form of 1,283 rows x 29,205 features is 300 MB in
        # one allocation. The guard is that evaluation goes through the serving
        # construction, not that it does so in a single call.
        self.assertTrue("predict_proba_texts(" in source
                        or "make_prediction_features_batch(" in source)
        self.assertNotIn("vectorizer.transform(test_df", source)

    def test_the_chunked_scorer_really_does_delegate(self):
        """Otherwise the guard above could be satisfied by a name alone."""
        source = (SERVICE_DIR / "binary_truth_mlp.py").read_text()
        body = source.split("def predict_proba_texts(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("make_prediction_features_batch(", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestReportedUncertainty(unittest.TestCase):
    """A point estimate with no interval invites chasing noise.

    The corrected accuracy (0.6188) sits only 5.5 points above the majority
    baseline on 1267 rows. Whether that gap is real is the whole question, and
    it cannot be answered by the point estimate alone.
    """

    def setUp(self):
        import evaluate_production_model as epm
        self.epm = epm

    def test_a_perfect_predictor_has_a_tight_interval_at_one(self):
        y = np.array([1, 0] * 100)
        low, high = self.epm.bootstrap_interval(y, y.copy(), resamples=200)
        self.assertEqual((low, high), (1.0, 1.0))

    def test_the_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        pred = np.where(rng.random(500) < 0.7, y, 1 - y)
        accuracy = float((pred == y).mean())
        low, high = self.epm.bootstrap_interval(y, pred, resamples=500)
        self.assertLessEqual(low, accuracy)
        self.assertGreaterEqual(high, accuracy)

    def test_a_smaller_sample_gives_a_wider_interval(self):
        rng = np.random.default_rng(1)
        def width(n):
            y = rng.integers(0, 2, n)
            pred = np.where(rng.random(n) < 0.7, y, 1 - y)
            low, high = self.epm.bootstrap_interval(y, pred, resamples=400)
            return high - low
        self.assertGreater(width(100), width(2000))

    def test_it_is_reproducible(self):
        y = np.array([1, 0, 1, 1, 0] * 40)
        pred = np.array([1, 1, 1, 0, 0] * 40)
        first = self.epm.bootstrap_interval(y, pred, resamples=200, seed=7)
        second = self.epm.bootstrap_interval(y, pred, resamples=200, seed=7)
        self.assertEqual(first, second)


class TestCalibration(unittest.TestCase):
    """The score is shown to users and consumed as a prior, so whether it
    means what it says matters more than whether it is often right."""

    def setUp(self):
        import evaluate_production_model as epm
        self.epm = epm

    def test_a_perfectly_calibrated_model_scores_zero_error(self):
        """Half the 0.5-bin true, all of the 0.9-bin true."""
        probabilities = np.array([0.5] * 100 + [0.95] * 100)
        y = np.array([1] * 50 + [0] * 50 + [1] * 100)
        report = self.epm.calibration_report(y, probabilities)
        self.assertLess(report["expected_calibration_error"], 0.03)

    def test_an_overconfident_model_is_penalised(self):
        """Says 0.95, right half the time."""
        probabilities = np.array([0.95] * 100)
        y = np.array([1] * 50 + [0] * 50)
        report = self.epm.calibration_report(y, probabilities)
        self.assertGreater(report["expected_calibration_error"], 0.4)

    def test_empty_bins_are_omitted_rather_than_reported_as_zero(self):
        probabilities = np.array([0.55] * 20)
        report = self.epm.calibration_report(np.ones(20, dtype=int), probabilities)
        self.assertEqual(len(report["bins"]), 1)
        self.assertEqual(report["bins"][0]["n"], 20)

    def test_probabilities_of_exactly_one_are_counted(self):
        """A half-open top bin silently drops them."""
        report = self.epm.calibration_report(
            np.ones(10, dtype=int), np.ones(10))
        self.assertEqual(sum(b["n"] for b in report["bins"]), 10)
