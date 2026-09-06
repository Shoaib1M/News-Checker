"""
FILE PURPOSE:
Measure STANCE_THRESHOLD and STANCE_DOMINANCE against a labelled corpus, using
the real NLI model, so they are set from data rather than intuition.

    python stance_sweep.py
    python stance_sweep.py --show-errors

WHY THIS EXISTS:
Those two constants decide, for every document the system reads, whether it
counts as supporting the claim, contradicting it, or neither. They were chosen
by hand. Changing them by hand is how a fact-checker quietly starts giving
different answers, and nothing in the test suite would notice: the tests pin
the RULE, not the settings.

The two failures they trade off are not symmetric, and the sweep reports them
separately for that reason:

  - A threshold set too LOW invents positions. A document that merely mentions
    the subject is recorded as confirming or refuting the claim, and enough of
    those produce a confident verdict out of coverage that said nothing.

  - A threshold set too HIGH discards real positions, and the system reports
    "insufficient evidence" about a claim its sources actually addressed.

The first is worse. A wrong answer is worse than no answer, so the sweep
prints precision on each direction alongside recall, and the setting to prefer
is the one that keeps precision high across a PLATEAU rather than the one that
peaks — a peak that a 0.02 change falls off is a fit to this corpus, not a
property of the model.

WHY IT IS NOT PART OF THE TEST SUITE:
It needs `transformers`, `torch`, and a model download, so it cannot run
offline. The corpus below is written out as text, not as scores: what the
model outputs for these pairs is what is being measured, and hardcoding scores
would measure nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from evidence_pipeline import (  # noqa: E402
    STANCE_DOMINANCE,
    STANCE_THRESHOLD,
    decide_stance,
)
from nli_service import get_nli_service  # noqa: E402


# (claim, passage, true stance)
#
# Written to cover the shapes that actually reach NLI in this pipeline, not
# textbook entailment pairs: paraphrase, syndication wording, a fact-check
# quoting the claim it refutes, a partial match, and coverage that is about
# the same subject while taking no position at all. The "neutral" rows are the
# ones that matter most — they are the majority of what retrieval returns.
CORPUS: list[tuple[str, str, str]] = [
    # ── Plain support ────────────────────────────────────────────────
    ("The prime minister of India resigned",
     "India's prime minister resigned on Tuesday after coalition talks collapsed.",
     "supports"),
    ("The prime minister of India resigned",
     "The Indian premier stepped down following a party revolt, his office said.",
     "supports"),
    ("Google was fined by the European Commission",
     "The European Commission fined Google over its search practices.",
     "supports"),
    ("A magnitude 7 earthquake struck Japan",
     "A magnitude 7.0 earthquake hit northern Japan early on Sunday.",
     "supports"),
    ("The central bank raised interest rates",
     "Policymakers voted to increase the benchmark rate by a quarter point.",
     "supports"),
    ("Apple announced a foldable iPhone",
     "Apple unveiled its first foldable iPhone at an event in Cupertino.",
     "supports"),

    # ── Plain contradiction ──────────────────────────────────────────
    ("The United States banned Google in all its cities",
     "No such prohibition exists and no bill has been introduced.",
     "contradicts"),
    ("The prime minister of India resigned",
     "The prime minister rejected calls to resign and will remain in office.",
     "contradicts"),
    ("The central bank raised interest rates",
     "The central bank left interest rates unchanged at its March meeting.",
     "contradicts"),
    ("A new law bans petrol cars from 2030",
     "The proposed ban was voted down and no restriction takes effect in 2030.",
     "contradicts"),
    ("The company filed for bankruptcy",
     "The company said it is profitable and has no plans to seek protection.",
     "contradicts"),

    # ── The fact-check shape: quotes the claim, then refutes it ──────
    ("The United States banned Google in all its cities",
     "Posts claim the United States banned Google in all its cities. "
     "This is false.",
     "contradicts"),
    ("Vaccines cause autism",
     "The claim that vaccines cause autism has been repeatedly debunked by "
     "large studies.",
     "contradicts"),

    # ── Same subject, no position — the majority of retrieval ────────
    ("The prime minister of India resigned",
     "The prime minister opened a new rail link in Chennai on Monday.",
     "neutral"),
    ("The prime minister of India resigned",
     "Delhi residents gave mixed reactions as crowds gathered outside "
     "parliament.",
     "neutral"),
    ("Google was fined by the European Commission",
     "Google expanded its advertising tools across the United States this "
     "quarter.",
     "neutral"),
    ("Google was fined by the European Commission",
     "The European Commission published its annual competition report.",
     "neutral"),
    ("The central bank raised interest rates",
     "Analysts are divided over what the central bank will do next month.",
     "neutral"),
    ("A magnitude 7 earthquake struck Japan",
     "Japan runs regular earthquake drills in schools and offices.",
     "neutral"),
    ("Apple announced a foldable iPhone",
     "Apple's share price closed slightly lower on Thursday.",
     "neutral"),
    ("The company filed for bankruptcy",
     "The company was founded in 1994 and employs 4,000 people.",
     "neutral"),

    # ── Partial: the subject matches, the assertion does not ─────────
    ("The prime minister of India resigned this morning",
     "The finance minister resigned this morning, the ministry confirmed.",
     "neutral"),
    ("Google was fined 5 billion dollars",
     "Regulators are considering whether to open an investigation into Google.",
     "neutral"),
]

GRID_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
GRID_DOMINANCE = [1.2, 1.4, 1.6, 1.8, 2.0]


def score_corpus() -> list[tuple[str, str, str, float, float]]:
    """Run the real model once over every pair; the sweep reuses the scores."""
    service = get_nli_service()
    scored = []
    for claim, passage, label in CORPUS:
        (result,) = service.score_many(claim, [passage])
        if not result.get("available"):
            raise SystemExit(
                "The NLI model is not available, so there is nothing to measure.\n"
                "Install it with:  pip install transformers torch\n"
                f"Model: {service.status.get('model')}  "
                f"error: {service.status.get('error')}"
            )
        scored.append((claim, passage, label,
                       result["entailment"], result["contradiction"]))
    return scored


# The corpus calls a document that takes no position "neutral"; the rule calls
# that outcome "unclear". They are the same class under two names, and scoring
# them as different ones counted every correct neutral row as an error — which
# understates accuracy exactly where the corpus is densest.
_LABEL_ALIASES = {"neutral": "unclear"}


def _canonical(label: str) -> str:
    return _LABEL_ALIASES.get(label, label)


def evaluate(scored, threshold: float, dominance: float) -> dict:
    """Per-direction precision and recall at one setting."""
    counts = {d: {"tp": 0, "fp": 0, "fn": 0} for d in ("supports", "contradicts")}
    correct = 0
    for _claim, _passage, raw_truth, entail, contradict in scored:
        truth = _canonical(raw_truth)
        predicted = decide_stance(entail, contradict, threshold, dominance)
        if predicted == truth:
            correct += 1
        for direction in counts:
            if predicted == direction and truth != direction:
                counts[direction]["fp"] += 1
            elif predicted == direction and truth == direction:
                counts[direction]["tp"] += 1
            elif predicted != direction and truth == direction:
                counts[direction]["fn"] += 1

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    report = {"accuracy": correct / len(scored)}
    for direction, c in counts.items():
        report[direction] = {
            "precision": ratio(c["tp"], c["tp"] + c["fp"]),
            "recall": ratio(c["tp"], c["tp"] + c["fn"]),
            "invented": c["fp"],   # positions the system made up
            "missed": c["fn"],     # positions it failed to see
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep the stance thresholds.")
    parser.add_argument("--show-errors", action="store_true",
                        help="print every pair the current setting gets wrong")
    args = parser.parse_args()

    print(f"\nScoring {len(CORPUS)} labelled pairs with the real model...\n")
    scored = score_corpus()

    print(f"{'thresh':>7} {'domin':>6} {'acc':>6} "
          f"{'sup P':>6} {'sup R':>6} {'con P':>6} {'con R':>6} "
          f"{'invented':>9}")
    print("-" * 62)
    for dominance in GRID_DOMINANCE:
        for threshold in GRID_THRESHOLDS:
            r = evaluate(scored, threshold, dominance)
            invented = r["supports"]["invented"] + r["contradicts"]["invented"]
            marker = "  <- current" if (
                abs(threshold - STANCE_THRESHOLD) < 1e-9
                and abs(dominance - STANCE_DOMINANCE) < 1e-9
            ) else ""
            print(f"{threshold:>7.2f} {dominance:>6.1f} {r['accuracy']:>6.2f} "
                  f"{r['supports']['precision']:>6.2f} {r['supports']['recall']:>6.2f} "
                  f"{r['contradicts']['precision']:>6.2f} "
                  f"{r['contradicts']['recall']:>6.2f} {invented:>9}{marker}")
        print()

    print("'invented' counts documents recorded as taking a position they do "
          "not take.\nPrefer a setting in the middle of a stable region over "
          "one that peaks:\na peak a 0.02 step falls off is a fit to these 23 "
          "pairs, not to the model.\n")

    if args.show_errors:
        print(f"Errors at the current setting "
              f"({STANCE_THRESHOLD}, {STANCE_DOMINANCE}):\n")
        for claim, passage, raw_truth, entail, contradict in scored:
            truth = _canonical(raw_truth)
            predicted = decide_stance(entail, contradict)
            if predicted != truth:
                print(f"  expected {truth:<11} got {predicted:<11} "
                      f"(e={entail:.2f} c={contradict:.2f})")
                print(f"    claim  : {claim}")
                print(f"    passage: {passage[:88]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
