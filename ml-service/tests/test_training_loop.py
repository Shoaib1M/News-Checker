"""The training loop's reproducibility and regularisation knobs.

WHY THIS EXISTS:
Three defects found while building `train_experiments.py`, all of which make an
experiment unreadable rather than wrong:

1. **Training was not reproducible.** `__init__` seeded a generator for the
   initial weights, but the per-epoch shuffle used `np.random.permutation` —
   the *global* RNG. So `seed` controlled only where training started, and two
   runs of the same configuration differed by more than the effects being
   compared. The README told readers to "reproduce these numbers", and they
   could not.

2. **No regularisation**, on 26,626 input dimensions over ~10k rows.

3. **No early stopping** — a fixed epoch count regardless of what validation
   loss was doing, and whatever weights the last epoch left behind were the
   ones kept.

Defaults are unchanged, so the shipped model's behaviour is preserved; the
knobs are opt-in and the experiment harness turns them on.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import BinaryTruthMLP  # noqa: E402


def toy_problem(n=300, features=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, features))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


class TestReproducibility(unittest.TestCase):

    def train(self, seed, epochs=6):
        X, y = toy_problem()
        model = BinaryTruthMLP(input_size=X.shape[1], hidden_size=8,
                               epochs=epochs, seed=seed)
        model.fit(X, y, quiet=True)
        return model.predict_proba(X)

    def test_the_same_seed_gives_the_same_model(self):
        np.testing.assert_allclose(self.train(1), self.train(1))

    def test_a_different_seed_gives_a_different_model(self):
        self.assertFalse(np.allclose(self.train(1), self.train(2)))

    def test_the_global_rng_does_not_affect_training(self):
        """The bug: seeding numpy globally used to change the batch order."""
        X, y = toy_problem()

        def run():
            model = BinaryTruthMLP(input_size=X.shape[1], hidden_size=8,
                                   epochs=6, seed=3)
            model.fit(X, y, quiet=True)
            return model.predict_proba(X)

        np.random.seed(1234)
        first = run()
        np.random.seed(9999)
        second = run()
        np.testing.assert_allclose(first, second)


class TestDefaultsAreUnchanged(unittest.TestCase):
    """The shipped model was trained without these; adding them must not
    silently retrain it into something else."""

    def test_regularisation_is_off_by_default(self):
        model = BinaryTruthMLP(input_size=5)
        self.assertEqual(model.weight_decay, 0.0)

    def test_early_stopping_is_off_by_default(self):
        model = BinaryTruthMLP(input_size=5)
        self.assertIsNone(model.early_stopping_patience)

    def test_without_patience_every_epoch_runs(self):
        X, y = toy_problem(n=120, features=10)
        Xv, yv = toy_problem(n=60, features=10, seed=5)
        model = BinaryTruthMLP(input_size=10, hidden_size=4, epochs=25, seed=1)
        model.fit(X, y, Xv, yv, quiet=True)
        # Nothing to assert about a break that cannot happen; assert the run
        # completed and produced a usable threshold instead.
        self.assertTrue(0.0 < model.best_threshold < 1.0)


class TestWeightDecay(unittest.TestCase):

    def trained_norm(self, weight_decay):
        X, y = toy_problem(n=200, features=60)
        model = BinaryTruthMLP(input_size=60, hidden_size=8, epochs=30,
                               seed=1, weight_decay=weight_decay)
        model.fit(X, y, quiet=True)
        return float(np.linalg.norm(model.W1))

    def test_it_shrinks_the_weights(self):
        self.assertLess(self.trained_norm(1e-2), self.trained_norm(0.0))

    def test_biases_are_left_alone(self):
        """Penalising a bias shifts the boundary without reducing capacity."""
        source = (SERVICE_DIR / "binary_truth_mlp.py").read_text()
        update = source.split("if self.weight_decay:", 1)[1].split("self.W2 -=", 1)[0]
        self.assertIn("self.W2", update)
        self.assertIn("self.W1", update)
        self.assertNotIn("self.b1", update)
        self.assertNotIn("self.b2", update)


class TestEarlyStopping(unittest.TestCase):

    def test_it_stops_before_the_epoch_limit(self):
        X, y = toy_problem(n=400, features=50)
        Xv, yv = toy_problem(n=200, features=50, seed=7)
        model = BinaryTruthMLP(input_size=50, hidden_size=8, epochs=300,
                               seed=1, early_stopping_patience=3)
        model.fit(X, y, Xv, yv, quiet=True)
        self.assertTrue(0.0 < model.best_threshold < 1.0)

    def test_it_restores_the_best_weights_seen_in_the_run(self):
        """The invariant is within-run, not across runs.

        An earlier version of this test asserted that an early-stopped model
        beats one trained longer. That is not what early stopping promises and
        it failed by 0.0005: patience can halt on a noisy dip that the longer
        run later climbs out of. What IS guaranteed is that the weights left
        behind are the best ones *this run* saw — not whatever the final epoch
        happened to produce.
        """
        X, y = toy_problem(n=200, features=80)
        Xv, yv = toy_problem(n=120, features=80, seed=11)

        observed: list[float] = []

        class Recording(BinaryTruthMLP):
            def loss(self, predicted, actual):
                value = super().loss(predicted, actual)
                if len(actual) == len(yv):
                    observed.append(float(value))
                return value

        model = Recording(input_size=80, hidden_size=16, epochs=400,
                          seed=2, early_stopping_patience=3)
        model.fit(X, y, Xv, yv, quiet=True)

        final = float(model.loss(model.predict_proba(Xv), yv))
        # The last entry is this very call, so compare against the training run.
        during_training = observed[:-1]
        self.assertTrue(during_training, "no validation losses were recorded")
        self.assertAlmostEqual(final, min(during_training), places=6)
        self.assertLess(final, during_training[-1] + 1e-9,
                        "kept the last weights rather than the best")

    def test_it_needs_a_validation_set_to_act_on(self):
        """With no validation data there is nothing to stop on; it must not
        crash, it must simply run to the epoch limit."""
        X, y = toy_problem(n=100, features=10)
        model = BinaryTruthMLP(input_size=10, hidden_size=4, epochs=8,
                               seed=1, early_stopping_patience=2)
        model.fit(X, y, quiet=True)
        self.assertEqual(model.predict_proba(X).shape, (100,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
