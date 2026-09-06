"""The vocabulary's column order must not depend on the process.

WHY THIS EXISTS:
`build_vocab` counted document frequencies by iterating a `set` of token
strings. Python randomises string hashing per process, so the insertion order
of the counting dict differed between runs — and step 2 assigned each token its
column index in exactly that order.

The consequence was not a wrong answer but an unmeasurable one: the feature
matrix was column-permuted per process, so seeded weight initialisation lined
up against different tokens, and two runs of an identical training
configuration produced different models. It was invisible from inside a single
interpreter, which is why an earlier reproducibility test passed while the
defect was live. It surfaced only as the same sweep variant scoring 0.6402 in
one invocation and 0.6379 in the next.

These tests run the vectorizer in SEPARATE interpreters with different
`PYTHONHASHSEED` values, because that is the only way to see it.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from tfidf import TFIDFVectorizer  # noqa: E402

DOCUMENTS = [
    "the prime minister of India resigned this morning",
    "the central bank raised interest rates by 25 basis points",
    "regulators approved the merger between the two carriers",
    "a magnitude 7 earthquake struck northern Japan",
    "shares soared after the earnings report was published",
    "the minister was sacked after the parliamentary vote",
]

PROBE = """
import json, sys
sys.path.insert(0, {service!r})
from tfidf import TFIDFVectorizer
docs = json.loads({docs!r})
v = TFIDFVectorizer()
v.build_vocab(docs)
print(json.dumps(list(v.vocab.items())))
"""


def vocab_under_hash_seed(seed: str):
    import json
    code = PROBE.format(service=str(SERVICE_DIR), docs=json.dumps(DOCUMENTS))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr[-800:])
    return json.loads(result.stdout)


TRAIN_PROBE = """
import json, sys
sys.path.insert(0, {service!r})
import numpy as np
from tfidf import TFIDFVectorizer
from binary_truth_mlp import BinaryTruthMLP, HISTORY_COLUMNS, make_prediction_features_batch
docs = json.loads({docs!r})
labels = np.array([0, 1] * (len(docs) // 2))
v = TFIDFVectorizer(min_df=1)
v.build_vocab(docs)
X = make_prediction_features_batch(v, np.ones((1, len(HISTORY_COLUMNS))), docs)
m = BinaryTruthMLP(input_size=X.shape[1], hidden_size=4, epochs=3, seed=7)
m.fit(X, labels, quiet=True)
print(json.dumps([round(float(p), 10) for p in m.predict_proba(X)]))
"""


def predictions_under_hash_seed(seed: str):
    import json
    code = TRAIN_PROBE.format(service=str(SERVICE_DIR), docs=json.dumps(DOCUMENTS))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr[-800:])
    return json.loads(result.stdout)


class TestTrainingIsReproducibleAcrossProcesses(unittest.TestCase):
    """What the column-order bug actually cost.

    Seeded weights lined up against different tokens each run, so an identical
    configuration trained into a different model. The sweep saw the same
    variant score 0.6402 and then 0.6379, which is larger than most of the
    effects it exists to measure — every comparison it made was unreadable.
    """

    def test_the_same_seed_trains_the_same_model_in_any_process(self):
        self.assertEqual(predictions_under_hash_seed("0"),
                         predictions_under_hash_seed("54321"))


class TestColumnOrderIsCanonical(unittest.TestCase):

    def test_two_hash_seeds_give_the_same_columns(self):
        """The bug, seen the only way it can be seen."""
        first = vocab_under_hash_seed("0")
        second = vocab_under_hash_seed("12345")
        self.assertEqual(first, second,
                         "vocabulary column order depends on the hash seed")

    def test_a_third_seed_agrees_too(self):
        self.assertEqual(vocab_under_hash_seed("1"), vocab_under_hash_seed("99999"))

    def test_indices_are_dense_and_start_at_zero(self):
        vectorizer = TFIDFVectorizer()
        vectorizer.build_vocab(DOCUMENTS)
        indices = sorted(vectorizer.vocab.values())
        self.assertEqual(indices, list(range(len(indices))))
        self.assertEqual(vectorizer.vocab_size, len(indices))

    def test_tokens_are_assigned_in_sorted_order(self):
        """A canonical order, so a rebuilt vocabulary matches a saved one."""
        vectorizer = TFIDFVectorizer()
        vectorizer.build_vocab(DOCUMENTS)
        tokens = [t for t, _i in sorted(vectorizer.vocab.items(), key=lambda kv: kv[1])]
        self.assertEqual(tokens, sorted(tokens))

    def test_min_df_still_filters(self):
        """Sorting must not change WHICH tokens survive, only their order."""
        loose = TFIDFVectorizer(min_df=1)
        strict = TFIDFVectorizer(min_df=3)
        loose.build_vocab(DOCUMENTS)
        strict.build_vocab(DOCUMENTS)
        self.assertLess(strict.vocab_size, loose.vocab_size)
        self.assertTrue(set(strict.vocab) <= set(loose.vocab))


if __name__ == "__main__":
    unittest.main(verbosity=2)
