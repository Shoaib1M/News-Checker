"""The LIAR splits must be read whole.

WHY THIS EXISTS:
Every `pd.read_csv(path, sep="\\t", names=COLUMNS)` in this project was reading
fewer rows than the file contains. pandas defaults to `quotechar='"'`; LIAR is
a tab-separated file full of quoted political speech, so one `"` in a claim
opened a field that stayed open until the next `"` — several lines later —
and everything in between was absorbed into that row's `statement` under the
FIRST row's label.

    split   lines in file   rows the old call returned   lost
    train      10,269                 10,240             29
    valid       1,284                  1,284              0
    test        1,283                  1,267             16

So the "1,267 LIAR test examples" this project reported was the bug, not the
dataset, and two of those rows were several claims glued together with raw tab
characters and other rows' metadata inside them — the longest "statement"
pandas produced was 431 words against a true maximum of 48.

None of it is visible from the outside: no warning, no exception, and an
accuracy figure that looks perfectly reasonable. Hence these tests.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import COLUMNS, DATA_DIR, load_split  # noqa: E402

EXPECTED_ROWS = {"train": 10269, "valid": 1284, "test": 1283}
RATINGS = {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}


class TestEveryRowIsRead(unittest.TestCase):

    def test_row_counts_match_the_files_on_disk(self):
        for split, expected in EXPECTED_ROWS.items():
            with self.subTest(split=split):
                path = DATA_DIR / f"{split}.tsv"
                with path.open(encoding="utf-8") as handle:
                    lines = sum(1 for _ in handle)
                self.assertEqual(lines, expected, "the data file itself changed")
                self.assertEqual(len(load_split(split)), expected,
                                 f"{split}.tsv lost rows on the way in")

    def test_no_statement_contains_a_tab(self):
        """A tab inside a statement means the parser merged fields."""
        for split in EXPECTED_ROWS:
            with self.subTest(split=split):
                statements = load_split(split)["statement"].fillna("").astype(str)
                offenders = [s for s in statements if "\t" in s]
                self.assertEqual(offenders, [], "rows were glued together")

    def test_no_statement_is_absurdly_long(self):
        """LIAR claims are one sentence. 431 words is several rows in a trench
        coat; the real maximum across all three splits is 48."""
        for split in EXPECTED_ROWS:
            with self.subTest(split=split):
                statements = load_split(split)["statement"].fillna("").astype(str)
                self.assertLess(max(len(s.split()) for s in statements), 80)

    def test_every_row_carries_one_of_the_six_ratings(self):
        """A merged row can also shift the columns, which shows up as a label
        that is not a rating at all."""
        for split in EXPECTED_ROWS:
            with self.subTest(split=split):
                labels = set(load_split(split)["label"].dropna())
                self.assertEqual(labels - RATINGS, set())

    def test_the_columns_are_the_ones_the_loader_names(self):
        frame = load_split("valid")
        self.assertEqual(list(frame.columns), COLUMNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
