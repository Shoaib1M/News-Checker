"""The credit-history feature leaks the target unless the current row is removed.

WHY THIS EXISTS:
LIAR ships five "credit history" counts per statement — how often that speaker
has been rated barely-true, false, half-true, mostly-true and pants-fire. They
read as the speaker's *past* record. They include **the current statement's own
label**.

Measured over the 2,054 speakers who appear exactly once in the whole dataset:

    own-label count == 1    99.2%
    total history  == 1     98.9%
    total history  == 0      0.8%

A speaker with a single statement carries exactly one count, sitting in that
statement's own label column. The feature partly *is* the target, and any
accuracy trained on it is inflated. This project's old "72.38% on LIAR" came
from exactly that, and the README warned the number was metadata-dependent
without anyone naming it as leakage.

There is a second-order version. LIAR has no `true` column, so a solo speaker
labelled "true" has all-zero history — 352 of the 353 such rows. All-zero
history was therefore itself a signal for "true". Subtracting the current row
removes both leaks: a solo "false" speaker becomes all-zero too.

These tests run against the real dataset, because the leak is a property of the
data rather than of the code.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import (  # noqa: E402
    COLUMNS,
    HISTORY_COLUMNS,
    LABEL_TO_HISTORY_COLUMN,
    build_history_features,
)

DATA = SERVICE_DIR / "data"


def load_all():
    frames = [pd.read_csv(DATA / f"{name}.tsv", sep="\t", names=COLUMNS)
              for name in ("train", "valid", "test")]
    return pd.concat(frames, ignore_index=True)


@unittest.skipUnless((DATA / "train.tsv").exists(), "LIAR splits not present")
class TestTheLeakIsReal(unittest.TestCase):
    """Documents the defect, so removing the fix cannot look harmless."""

    @classmethod
    def setUpClass(cls):
        rows = load_all()
        counts = rows["speaker"].value_counts()
        cls.solo = rows[rows["speaker"].isin(counts[counts == 1].index)].reset_index(drop=True)

    def test_a_speaker_with_one_statement_still_has_history(self):
        raw, _ = build_history_features(self.solo)
        carrying = float((raw.sum(axis=1) > 0).mean())
        self.assertGreater(
            carrying, 0.5,
            "expected most one-statement speakers to carry a count they cannot "
            "have earned — if this fails the dataset changed",
        )

    def test_that_count_sits_in_their_own_label_s_column(self):
        labelled = self.solo[self.solo["label"].isin(LABEL_TO_HISTORY_COLUMN)]
        own = labelled.apply(
            lambda r: r[LABEL_TO_HISTORY_COLUMN[r["label"]]], axis=1)
        self.assertGreater(float((own >= 1).mean()), 0.95)


@unittest.skipUnless((DATA / "train.tsv").exists(), "LIAR splits not present")
class TestDeleakingRemovesIt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rows = load_all()
        counts = rows["speaker"].value_counts()
        cls.solo = rows[rows["speaker"].isin(counts[counts == 1].index)].reset_index(drop=True)

    def test_solo_speakers_end_up_with_no_history(self):
        """They have no OTHER statements, so their history must be empty."""
        clean, _ = build_history_features(
            self.solo, deleak_labels=self.solo["label"])
        carrying = float((clean.sum(axis=1) > 0).mean())
        self.assertLess(carrying, 0.02, f"{carrying:.1%} still carry history")

    def test_the_true_label_asymmetry_is_gone(self):
        """All-zero history was itself a signal for "true"; after de-leaking a
        solo "false" speaker is all-zero too, so it signals nothing."""
        clean, _ = build_history_features(
            self.solo, deleak_labels=self.solo["label"])
        empty = clean.sum(axis=1) == 0
        share_true = float((self.solo.loc[empty, "label"] == "true").mean())
        self.assertLess(
            share_true, 0.35,
            "empty history is still dominated by one label, so it still leaks",
        )

    def test_no_count_goes_negative(self):
        """A few rows carry a zero count for their own label already."""
        clean, _ = build_history_features(
            self.solo, deleak_labels=self.solo["label"])
        self.assertGreaterEqual(float(clean.min()), 0.0)

    def test_a_multi_statement_speaker_keeps_the_rest_of_their_record(self):
        rows = load_all()
        counts = rows["speaker"].value_counts()
        busy = rows[rows["speaker"].isin(counts[counts > 20].index)].reset_index(drop=True)
        raw, _ = build_history_features(busy)
        clean, _ = build_history_features(busy, deleak_labels=busy["label"])
        self.assertGreater(float((clean.sum(axis=1) > 0).mean()), 0.95)
        self.assertLess(float(clean.sum()), float(raw.sum()),
                        "de-leaking must remove something")


class TestInferenceIsUnaffected(unittest.TestCase):
    """A live claim is in nobody's history, so there is nothing to subtract."""

    def test_no_labels_means_no_subtraction(self):
        frame = pd.DataFrame([{c: 5 for c in HISTORY_COLUMNS}])
        untouched, _ = build_history_features(frame)
        expected = np.log1p(np.full((1, len(HISTORY_COLUMNS)), 5.0))
        np.testing.assert_allclose(untouched * np.maximum(expected.max(), 1),
                                   expected, rtol=1e-6)

    def test_the_serving_path_passes_no_labels(self):
        """make_prediction_features_batch must never de-leak."""
        source = (SERVICE_DIR / "binary_truth_mlp.py").read_text()
        body = source.split("def make_prediction_features_batch", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("build_history_features(frame, train_max_values)", body)
        self.assertNotIn("deleak", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
