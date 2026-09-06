"""
FILE PURPOSE:
Train variants of the Binary Truth MLP and pick one, honestly.

    python train_experiments.py                 # sweep, report, recommend
    python train_experiments.py --save          # also write the winner to disk

WHY THIS EXISTS:
The shipped model's hyperparameters were chosen once and never revisited, and
the training loop had no regularisation and no early stopping on 26,626 input
dimensions over ~10k rows — the textbook setup for overfitting. Whether that
costs anything is an empirical question, and this answers it.

THE RULE THIS SCRIPT ENFORCES:
**Variants are selected on VALIDATION. The test set is touched once, by the
winner, at the end.**

That is not bureaucracy. Running six variants and shipping whichever scored
best on test is overfitting the test set with extra steps — the reported number
then describes the sweep, not the model, and it will not survive contact with
new data. `evaluate_production_model.py` reports the shipped model's test
metrics; this script's job is to decide *which* model that should be.

Every configuration is trained from the same seed, and the seed now controls
batch shuffling as well as weight initialisation, so two runs of the same
config give the same answer. Before that fix, `np.random.permutation` used the
global RNG and repeat runs differed by more than the effects being compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import (  # noqa: E402
    COLUMNS,
    HISTORY_COLUMNS,
    BinaryTruthMLP,
    find_best_threshold,
    labels_to_binary,
    make_prediction_features_batch,
    save_artifacts,
)
from tfidf import TFIDFVectorizer  # noqa: E402

# (name, vectorizer kwargs, model kwargs)
#
# Deliberately small. Each variant tests one idea against the baseline, so a
# difference can be attributed; a grid of everything-against-everything on
# 1284 validation rows would mostly measure noise.
# Every variant gets the SAME 70-epoch budget as the incumbent. That is a
# controlled comparison, not a time saving: at a larger cap, a variant that
# scored higher might simply have trained longer, and the knob being tested
# would not be what the number measured. Early stopping can still halt sooner —
# and if it never does, that is a finding in itself.
#
# The first run of "early stopping" used a 150-epoch cap and reached it without
# patience ever triggering, which says validation loss was still improving at
# epoch 150. That is worth knowing: this model is not overfitting as quickly as
# 26,626 dimensions over 10,240 rows would suggest.
VARIANTS = [
    ("baseline (shipped)",      {}, {}),
    ("early stopping",          {}, {"early_stopping_patience": 5, "epochs": 70}),
    ("L2 1e-4 + early stop",    {}, {"weight_decay": 1e-4, "early_stopping_patience": 5, "epochs": 70}),
    ("L2 1e-3 + early stop",    {}, {"weight_decay": 1e-3, "early_stopping_patience": 5, "epochs": 70}),
    ("hidden 32 + early stop",  {}, {"hidden_size": 32, "early_stopping_patience": 5, "epochs": 70}),
    ("hidden 128 + early stop", {}, {"hidden_size": 128, "early_stopping_patience": 5, "epochs": 70}),
    ("min_df 3 + early stop",   {"min_df": 3}, {"early_stopping_patience": 5, "epochs": 70}),
    ("min_df 1 + early stop",   {"min_df": 1}, {"early_stopping_patience": 5, "epochs": 70}),
    # Learning rate, added after a probe showed the shipped 0.05 leaves
    # validation loss at ln(2) with an output spread of 0.057 after 15 epochs —
    # the network is barely differentiating anything. At lr 0.5 the same 15
    # epochs reach the accuracy the incumbent needs 70 for.
    ("lr 0.25",                 {}, {"learning_rate": 0.25}),
    ("lr 0.5",                  {}, {"learning_rate": 0.5}),
    ("lr 1.0",                  {}, {"learning_rate": 1.0}),
    ("lr 0.5 + L2 1e-4",        {}, {"learning_rate": 0.5, "weight_decay": 1e-4}),
    # The combination the lr probe points at: reach the good region fast, then
    # stop before overfitting undoes it. lr 0.5 scores 0.6332 at 15 epochs and
    # 0.6215 by 70, so the peak is real and the decline is real.
    ("lr 0.5 + early stop",     {}, {"learning_rate": 0.5, "early_stopping_patience": 5}),
    ("lr 0.25 + early stop",    {}, {"learning_rate": 0.25, "early_stopping_patience": 5}),
    # One motivated combination, not a grid. Both min_df 3 (smaller vocabulary)
    # and early stopping point the same way — less effective capacity — so
    # pairing them tests that reading rather than fishing. Adding combinations
    # indefinitely would inflate the leader by multiple comparisons alone.
    ("min_df 3 + lr 0.5 + stop", {"min_df": 3},
     {"learning_rate": 0.5, "early_stopping_patience": 5}),
]

BASE_MODEL_KWARGS = {
    "hidden_size": 64,
    "learning_rate": 0.05,
    "epochs": 70,
    "batch_size": 128,
    "seed": 42,
}


def bootstrap_interval(y_true, predictions, resamples=1000, seed=0):
    rng = np.random.default_rng(seed)
    correct = (predictions == y_true).astype(float)
    n = len(correct)
    means = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_features(vectorizer_kwargs, splits):
    """Vocabulary from TRAIN only, then features for every split.

    Building the vocabulary on anything but train leaks the evaluation sets
    into the feature space, which inflates every number that follows.
    """
    train_text, valid_text, test_text = splits
    vectorizer = TFIDFVectorizer(**vectorizer_kwargs)
    vectorizer.build_vocab(train_text)
    train_max_values = np.ones((1, len(HISTORY_COLUMNS)))
    matrices = tuple(
        make_prediction_features_batch(vectorizer, train_max_values, text)
        for text in (train_text, valid_text, test_text)
    )
    return vectorizer, train_max_values, matrices


def choose(variants, incumbent_name="baseline (shipped)"):
    """Pick a winner on VALIDATION and say whether it is decisive.

    A variant only displaces the incumbent if it clears the incumbent's
    validation confidence interval. Anything inside that interval is noise, and
    swapping models on noise means retraining forever while the number wanders.
    """
    if not variants:
        raise SystemExit("no results to choose from — run some variants first.")
    winner = max(variants, key=lambda r: r["valid_accuracy"])
    incumbent = next((r for r in variants if r["name"] == incumbent_name), None)

    if incumbent is None:
        return winner, None, (
            f"{winner['name']} leads, but the shipped configuration was not "
            f"among the variants run, so there is nothing to compare against. "
            f"Run '{incumbent_name}' before concluding anything."
        )
    if winner["name"] == incumbent["name"]:
        return incumbent, incumbent, "The shipped configuration is already the best of these."
    if winner["valid_accuracy"] > incumbent["valid_ci"][1]:
        return winner, incumbent, (
            f"{winner['name']} clears the incumbent's validation interval "
            f"({incumbent['valid_ci'][1]:.4f}) — a real improvement.")
    return incumbent, incumbent, (
        f"{winner['name']} leads but sits INSIDE the incumbent's validation "
        f"interval [{incumbent['valid_ci'][0]:.4f}, {incumbent['valid_ci'][1]:.4f}] "
        f"— not distinguishable from noise. Keeping the shipped model.")


def report_from_file(path: Path) -> int:
    """Selection over accumulated results, without retraining.

    The test evaluation still happens exactly once, here, after selection — it
    is not stored per variant, precisely so that no run can quietly pick the
    configuration that happened to score best on held-out data.
    """
    if not path.exists():
        raise SystemExit(f"{path} does not exist — run some variants first.")
    payload = json.loads(path.read_text())
    variants = payload.get("variants", [])
    print(f"\n{len(variants)} variant(s) collected\n")
    print(f"{'variant':<26} {'valid acc':>10} {'95% CI':>18}")
    print("-" * 56)
    for r in sorted(variants, key=lambda r: -r["valid_accuracy"]):
        print(f"{r['name']:<26} {r['valid_accuracy']:>10.4f} "
              f"[{r['valid_ci'][0]:.4f}, {r['valid_ci'][1]:.4f}]")
    _shipped, _incumbent, verdict = choose(variants)
    print(f"\n{verdict}\n")
    print("To evaluate the chosen model on the held-out test set, rerun it with "
          "--only and let this script's live path do the single test pass.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="write the winning model over binary_truth_mlp.pkl")
    parser.add_argument("--out", default="experiment_results.json")
    parser.add_argument("--only", default="",
                        help="comma-separated variant names to run this time")
    parser.add_argument("--append", action="store_true",
                        help="merge into --out instead of overwriting")
    parser.add_argument("--report", action="store_true",
                        help="select and evaluate from --out without training")
    args = parser.parse_args()

    # Chunking exists because one variant costs 90-420s and a single run of all
    # eight does not fit in one sitting. Successive invocations accumulate into
    # --out; --report then does selection and the single test evaluation over
    # whatever has been collected.
    if args.report:
        return report_from_file(Path(args.out))

    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    if wanted:
        unknown = wanted - {name for name, _v, _m in VARIANTS}
        if unknown:
            raise SystemExit(f"unknown variant(s): {sorted(unknown)}\n"
                             f"available: {[n for n, _v, _m in VARIANTS]}")

    data = SERVICE_DIR / "data"
    frames = [pd.read_csv(data / f"{name}.tsv", sep="\t", names=COLUMNS)
              for name in ("train", "valid", "test")]
    splits = tuple(f["statement"].fillna("").astype(str) for f in frames)
    y_train, y_valid, y_test = (labels_to_binary(f["label"]) for f in frames)

    baseline = float(max(y_valid.mean(), 1 - y_valid.mean()))
    print(f"\ntrain {len(y_train)} · valid {len(y_valid)} · test {len(y_test)}")
    print(f"validation majority-class baseline: {baseline:.4f}\n")
    print(f"{'variant':<26} {'vocab':>7} {'valid acc':>10} {'95% CI':>16} "
          f"{'thresh':>7} {'secs':>6}")
    print("-" * 78)

    feature_cache: dict[str, tuple] = {}
    results = []

    for name, vectorizer_kwargs, model_kwargs in VARIANTS:
        if wanted and name not in wanted:
            continue
        cache_key = json.dumps(vectorizer_kwargs, sort_keys=True)
        if cache_key not in feature_cache:
            feature_cache[cache_key] = build_features(vectorizer_kwargs, splits)
        vectorizer, train_max_values, (X_train, X_valid, X_test) = feature_cache[cache_key]

        settings = {**BASE_MODEL_KWARGS, **model_kwargs}
        started = time.time()
        model = BinaryTruthMLP(input_size=X_train.shape[1], **settings)
        model.fit(X_train, y_train, X_valid, y_valid, quiet=True)
        elapsed = time.time() - started

        valid_scores = model.predict_proba(X_valid)
        threshold, valid_accuracy = find_best_threshold(valid_scores, y_valid)
        model.best_threshold = threshold
        low, high = bootstrap_interval(
            y_valid, (valid_scores >= threshold).astype(int))

        print(f"{name:<26} {X_train.shape[1]:>7} {valid_accuracy:>10.4f} "
              f"[{low:.4f}, {high:.4f}] {threshold:>7.2f} {elapsed:>6.0f}")

        results.append({
            "name": name, "valid_accuracy": round(float(valid_accuracy), 4),
            "valid_ci": [round(low, 4), round(high, 4)],
            "threshold": round(float(threshold), 4),
            "vocab_size": int(X_train.shape[1]), "seconds": round(elapsed, 1),
            "settings": settings, "vectorizer": vectorizer_kwargs,
            "_model": model, "_vectorizer": vectorizer,
            "_train_max_values": train_max_values, "_X_test": X_test,
        })

    # Merge with anything a previous invocation collected, so chunked runs
    # accumulate. Re-running a variant replaces its earlier entry.
    output_path = Path(args.out)
    previous = []
    if args.append and output_path.exists():
        previous = json.loads(output_path.read_text()).get("variants", [])
    fresh_names = {r["name"] for r in results}
    collected = [r for r in previous if r["name"] not in fresh_names] + [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ]

    shipped_summary, _incumbent, verdict = choose(collected)
    print(f"\n{verdict}")

    # The test set is touched exactly once, by whichever model we would ship —
    # and only if that model was trained in THIS run and is still in memory.
    shipped = next((r for r in results if r["name"] == shipped_summary["name"]), None)
    if shipped is None:
        print(f"\n'{shipped_summary['name']}' was chosen but trained in an "
              f"earlier invocation, so there is no model in memory to evaluate. "
              f"Rerun it with --only to get its held-out number.")
        output_path.write_text(json.dumps(
            {"validation_baseline": round(baseline, 4), "verdict": verdict,
             "shipped": shipped_summary["name"], "variants": collected}, indent=2))
        print(f"\nwritten to {output_path}")
        return 0

    model, X_test_matrix = shipped["_model"], shipped["_X_test"]
    test_predictions = (model.predict_proba(X_test_matrix) >= shipped["threshold"]).astype(int)
    test_accuracy = float((test_predictions == y_test).mean())
    test_low, test_high = bootstrap_interval(y_test, test_predictions)
    print(f"\nHELD-OUT TEST, {shipped['name']}: {test_accuracy:.4f} "
          f"[{test_low:.4f}, {test_high:.4f}]   (touched once, after selection)")

    payload = {
        "validation_baseline": round(baseline, 4),
        "verdict": verdict,
        "shipped": shipped["name"],
        "test_accuracy": round(test_accuracy, 4),
        "test_ci": [round(test_low, 4), round(test_high, 4)],
        "variants": collected,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten to {output_path}")

    if args.save and shipped["name"] != "baseline (shipped)":
        save_artifacts(SERVICE_DIR / "binary_truth_mlp.pkl", shipped["_model"],
                       shipped["_vectorizer"], shipped["_train_max_values"])
        print("saved the winner over binary_truth_mlp.pkl")
    elif args.save:
        print("nothing to save — the shipped model won.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
