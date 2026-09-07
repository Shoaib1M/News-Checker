"""The replacement claim model, the sparse feature path, and the artifact format.

WHY THIS EXISTS:
Three things changed in the claim classifier and each one can fail silently:

1. **Training moved to a sparse feature path.** Word + character TF-IDF is
   62,257 columns; the dense training matrix would be 6.1 GB, so `fit` reads
   the sparse triples while a live request still builds one dense row. Two
   code paths computing "the features" is exactly how the previous train/serve
   skew happened, so the equivalence is pinned here rather than assumed.

2. **The model is linear now, not an MLP.** A regularised linear model beat
   every hidden-layer configuration on cross-validation; see
   `docs/ML_MODEL_INVESTIGATION.md`. The old class is still present and still
   has to work, because the Model Comparison page reports it.

3. **The artifact format grew fields.** A vectorizer setting that the trainer
   saves and the loader defaults would serve a different feature space than it
   trained on — the same class of defect as the text skew, and just as quiet.
   Old artifacts must still load, and must still load as MLPs.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import (  # noqa: E402
    HISTORY_COLUMNS,
    BinaryTruthMLP,
    LinearTruthModel,
    build_feature_frame,
    build_text_input,
    load_artifacts,
    make_prediction_features,
    make_prediction_features_batch,
    make_training_features,
    predict_proba_texts,
    save_artifacts,
)
from tfidf import TFIDFVectorizer  # noqa: E402

CORPUS = [
    "the prime minister of India resigned this morning",
    "the central bank raised interest rates by 25 basis points",
    "Says the state budget doubled spending on schools in one year",
    "regulators approved the merger between the two carriers",
    "a magnitude 7 earthquake struck northern Japan",
    "taxes on taxpayers were raised by the taxman",
]


def fitted_vectorizer(**kwargs):
    settings = {"min_df": 1, "sublinear_tf": True, "smooth_idf": True}
    settings.update(kwargs)
    vectorizer = TFIDFVectorizer(**settings)
    vectorizer.build_vocab(build_text_input(build_feature_frame(CORPUS)))
    return vectorizer


class TestSparseMatrix(unittest.TestCase):
    """The two products a linear model needs, against the dense answer."""

    def setUp(self):
        self.vectorizer = fitted_vectorizer(char_ngram_range=(3, 5))
        self.matrix = self.vectorizer.transform_sparse(CORPUS, l2_normalize=False)
        self.dense = np.array([self.vectorizer.transform_one(d) for d in CORPUS])

    def test_it_holds_the_same_numbers_as_the_dense_form(self):
        np.testing.assert_allclose(self.matrix.to_dense(), self.dense)

    def test_dot_matches_a_dense_matrix_product(self):
        weights = np.random.default_rng(0).normal(size=self.matrix.n_columns)
        np.testing.assert_allclose(self.matrix.dot(weights), self.dense @ weights)

    def test_transpose_dot_matches_a_dense_gradient(self):
        per_row = np.random.default_rng(1).normal(size=len(CORPUS))
        np.testing.assert_allclose(self.matrix.transpose_dot(per_row),
                                   self.dense.T @ per_row)

    def test_l2_normalisation_gives_unit_rows(self):
        matrix = self.vectorizer.transform_sparse(CORPUS, l2_normalize=True)
        norms = np.bincount(matrix.rows, weights=matrix.values ** 2,
                            minlength=matrix.n_rows)
        np.testing.assert_allclose(norms, np.ones(len(CORPUS)))

    def test_a_document_with_no_known_tokens_does_not_crash(self):
        matrix = self.vectorizer.transform_sparse(["", "!!!"], l2_normalize=True)
        self.assertEqual(matrix.shape, (2, self.vectorizer.vocab_size))
        np.testing.assert_allclose(matrix.to_dense(), 0.0)


class TestTrainingAndServingAgree(unittest.TestCase):
    """The guarantee the whole design rests on."""

    def setUp(self):
        self.vectorizer = fitted_vectorizer(char_ngram_range=(3, 5))
        self.train_max_values = np.ones((1, len(HISTORY_COLUMNS)))

    def test_the_sparse_training_features_equal_the_dense_serving_features(self):
        dense = make_prediction_features_batch(
            self.vectorizer, self.train_max_values, CORPUS)
        sparse = make_training_features(self.vectorizer, CORPUS)
        self.assertEqual(dense.shape, sparse.shape)
        np.testing.assert_allclose(dense, sparse.to_dense(), atol=1e-12)

    def test_the_history_columns_are_present_but_empty_in_both(self):
        """Dropping them instead would shift every feature index by five."""
        sparse = make_training_features(self.vectorizer, CORPUS)
        self.assertEqual(sparse.n_columns,
                         self.vectorizer.vocab_size + len(HISTORY_COLUMNS))
        self.assertFalse((sparse.columns >= self.vectorizer.vocab_size).any())

    def test_a_model_scores_a_row_the_same_either_way(self):
        model = LinearTruthModel(
            input_size=self.vectorizer.vocab_size + len(HISTORY_COLUMNS))
        model.weights = np.random.default_rng(3).normal(size=model.input_size)
        model.bias = 0.25
        dense = make_prediction_features_batch(
            self.vectorizer, self.train_max_values, CORPUS)
        sparse = make_training_features(self.vectorizer, CORPUS)
        np.testing.assert_allclose(model.predict_proba(dense),
                                   model.predict_proba(sparse))

    def test_the_chunked_scorer_matches_scoring_in_one_go(self):
        model = LinearTruthModel(
            input_size=self.vectorizer.vocab_size + len(HISTORY_COLUMNS))
        model.weights = np.random.default_rng(4).normal(size=model.input_size)
        one_go = model.predict_proba(make_prediction_features_batch(
            self.vectorizer, self.train_max_values, CORPUS))
        chunked = predict_proba_texts(
            model, self.vectorizer, self.train_max_values, CORPUS, chunk=2)
        np.testing.assert_allclose(one_go, chunked)

    def test_a_single_row_matches_its_row_in_the_batch(self):
        batch = make_prediction_features_batch(
            self.vectorizer, self.train_max_values, CORPUS)
        for index, statement in enumerate(CORPUS):
            single = make_prediction_features(
                self.vectorizer, self.train_max_values, statement)
            np.testing.assert_allclose(single[0], batch[index])


class TestLinearTruthModel(unittest.TestCase):

    def separable_problem(self, n=400, features=60, seed=0):
        """A sparse problem shaped like the real one: L2-normalised rows with
        a handful of non-zeros each, and a label that one feature decides."""
        rng = np.random.default_rng(seed)
        rows, columns, values, labels = [], [], [], []
        for row in range(n):
            label = int(rng.random() < 0.5)
            picked = rng.choice(features, size=6, replace=False)
            if label:
                picked[0] = 0
            else:
                picked[0] = 1
            weight = rng.random(6) + 0.1
            weight /= np.linalg.norm(weight)
            rows.append(np.full(6, row)); columns.append(picked); values.append(weight)
            labels.append(label)
        from tfidf import SparseMatrix
        return SparseMatrix(np.concatenate(rows), np.concatenate(columns),
                            np.concatenate(values), n, features), np.array(labels, float)

    def test_it_learns_a_signal_the_features_contain(self):
        X, y = self.separable_problem()
        model = LinearTruthModel(input_size=X.n_columns, epochs=300)
        model.fit(X, y, quiet=True)
        self.assertGreater(float((model.predict_proba(X) >= 0.5).astype(int).__eq__(y).mean()),
                           0.9)

    def test_stronger_regularisation_shrinks_the_weights(self):
        X, y = self.separable_problem()
        norms = []
        for l2 in (1e-5, 1e-1):
            model = LinearTruthModel(input_size=X.n_columns, l2=l2, epochs=200)
            model.fit(X, y, quiet=True)
            norms.append(float(np.linalg.norm(model.weights)))
        self.assertLess(norms[1], norms[0])

    def test_the_penalty_leaves_the_bias_alone(self):
        """Penalising a bias drags the boundary towards 0.5 without reducing
        capacity — it is a worse model for no benefit."""
        source = (SERVICE_DIR / "binary_truth_mlp.py").read_text()
        body = source.split("class LinearTruthModel", 1)[1].split("\nclass ", 1)[0]
        update = body.split("gradient = ", 1)[1].split("step =", 1)[0]
        self.assertIn("self.l2 * self.weights", update)
        self.assertNotIn("self.l2 * self.bias", update)

    def test_training_is_deterministic(self):
        X, y = self.separable_problem()

        def run():
            model = LinearTruthModel(input_size=X.n_columns, epochs=40)
            model.fit(X, y, quiet=True)
            return model.predict_proba(X)

        np.testing.assert_allclose(run(), run())

    def test_the_threshold_comes_from_validation_not_training(self):
        X, y = self.separable_problem()
        Xv, yv = self.separable_problem(n=200, seed=9)
        model = LinearTruthModel(input_size=X.n_columns, epochs=60)
        model.fit(X, y, quiet=True)
        self.assertEqual(model.best_threshold, 0.5, "no validation set was given")
        model.fit(X, y, Xv, yv, quiet=True)
        self.assertTrue(0.0 < model.best_threshold < 1.0)


class TestArtifactFormat(unittest.TestCase):

    def round_trip(self, model, vectorizer):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pkl"
            save_artifacts(path, model, vectorizer, np.ones((1, len(HISTORY_COLUMNS))))
            return load_artifacts(path)

    def test_a_linear_model_survives_a_round_trip_unchanged(self):
        vectorizer = fitted_vectorizer(char_ngram_range=(3, 5))
        model = LinearTruthModel(
            input_size=vectorizer.vocab_size + len(HISTORY_COLUMNS))
        model.weights = np.random.default_rng(5).normal(size=model.input_size)
        model.bias = -0.3
        model.best_threshold = 0.53
        loaded, loaded_vectorizer, _ = self.round_trip(model, vectorizer)
        self.assertIsInstance(loaded, LinearTruthModel)
        self.assertEqual(loaded.best_threshold, 0.53)
        features = make_training_features(loaded_vectorizer, CORPUS)
        np.testing.assert_allclose(loaded.predict_proba(features),
                                   model.predict_proba(
                                       make_training_features(vectorizer, CORPUS)))

    def test_every_setting_that_changes_the_features_is_persisted(self):
        """A setting the trainer uses and the loader defaults would serve a
        different feature space than it trained on, silently."""
        vectorizer = fitted_vectorizer(char_ngram_range=(3, 6),
                                       sublinear_tf=True, smooth_idf=True)
        model = LinearTruthModel(
            input_size=vectorizer.vocab_size + len(HISTORY_COLUMNS))
        _, loaded, _ = self.round_trip(model, vectorizer)
        for attribute in ("ngram_range", "char_ngram_range", "min_df",
                          "sublinear_tf", "smooth_idf", "vocab_size"):
            self.assertEqual(getattr(loaded, attribute), getattr(vectorizer, attribute),
                             f"{attribute} did not survive the round trip")
        np.testing.assert_allclose(loaded.transform_one(CORPUS[0]),
                                   vectorizer.transform_one(CORPUS[0]))

    def test_an_artifact_written_before_the_linear_model_still_loads(self):
        """No "kind" key, no character/sublinear settings — every artifact in
        that format is an MLP trained on raw-TF word n-grams."""
        vectorizer = TFIDFVectorizer()
        vectorizer.build_vocab(CORPUS)
        mlp = BinaryTruthMLP(input_size=vectorizer.vocab_size + len(HISTORY_COLUMNS),
                             hidden_size=4)
        legacy = {
            "model": {k: v for k, v in mlp.state_dict().items() if k != "kind"},
            "vectorizer": {
                "vocab": vectorizer.vocab,
                "idf_values": vectorizer.idf_values,
                "vocab_size": vectorizer.vocab_size,
                "ngram_range": vectorizer.ngram_range,
                "min_df": vectorizer.min_df,
            },
            "train_max_values": np.ones((1, len(HISTORY_COLUMNS))),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            path.write_bytes(pickle.dumps(legacy))
            model, loaded, _ = load_artifacts(path)
        self.assertIsInstance(model, BinaryTruthMLP)
        self.assertIsNone(loaded.char_ngram_range)
        self.assertFalse(loaded.sublinear_tf)
        self.assertFalse(loaded.smooth_idf)


class TestTheShippedArtifact(unittest.TestCase):
    """Cheap checks on whatever is actually in binary_truth_mlp.pkl."""

    @classmethod
    def setUpClass(cls):
        path = SERVICE_DIR / "binary_truth_mlp.pkl"
        if not path.exists():
            raise unittest.SkipTest("no trained artifact on disk")
        cls.model, cls.vectorizer, cls.train_max_values = load_artifacts(path)

    def test_it_produces_a_probability_for_an_ordinary_claim(self):
        score = predict_proba_texts(
            self.model, self.vectorizer, self.train_max_values,
            ["the central bank raised interest rates by 25 basis points"])[0]
        self.assertTrue(0.0 < score < 1.0)

    def test_it_does_not_answer_the_same_thing_for_everything(self):
        """The model this replaced squeezed every score into [0.4, 0.7] and an
        earlier logistic-regression baseline into a 0.006-wide band, which is
        indistinguishable from having learned nothing."""
        scores = predict_proba_texts(
            self.model, self.vectorizer, self.train_max_values, CORPUS)
        self.assertGreater(float(scores.max() - scores.min()), 0.05)

    def test_the_threshold_is_a_real_decision_boundary(self):
        self.assertTrue(0.2 < self.model.best_threshold < 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
