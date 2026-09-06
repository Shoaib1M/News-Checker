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
VARIANTS = [
    ("baseline (shipped)",      {}, {}),
    ("early stopping",          {}, {"early_stopping_patience": 5, "epochs": 200}),
    ("L2 1e-4 + early stop",    {}, {"weight_decay": 1e-4, "early_stopping_patience": 5, "epochs": 200}),
    ("L2 1e-3 + early stop",    {}, {"weight_decay": 1e-3, "early_stopping_patience": 5, "epochs": 200}),
    ("hidden 32 + early stop",  {}, {"hidden_size": 32, "early_stopping_patience": 5, "epochs": 200}),
    ("hidden 128 + early stop", {}, {"hidden_size": 128, "early_stopping_patience": 5, "epochs": 200}),
    ("min_df 3 + early stop",   {"min_df": 3}, {"early_stopping_patience": 5, "epochs": 200}),
    ("min_df 1 + early stop",   {"min_df": 1}, {"early_stopping_patience": 5, "epochs": 200}),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="write the winning model over binary_truth_mlp.pkl")
    parser.add_argument("--out", default="experiment_results.json")
    args = parser.parse_args()

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

    winner = max(results, key=lambda r: r["valid_accuracy"])
    incumbent = results[0]
    print(f"\nbest on validation: {winner['name']}")

    # A variant only displaces the shipped model if it clears the incumbent's
    # interval. Anything inside it is noise, and swapping models on noise means
    # retraining forever while the number wanders.
    decisive = winner["valid_accuracy"] > incumbent["valid_ci"][1]
    if winner["name"] == incumbent["name"]:
        verdict = "The shipped configuration is already the best of these."
    elif decisive:
        verdict = (f"{winner['name']} clears the incumbent's validation interval "
                   f"({incumbent['valid_ci'][1]:.4f}) — a real improvement.")
    else:
        verdict = (f"{winner['name']} leads but sits INSIDE the incumbent's "
                   f"validation interval [{incumbent['valid_ci'][0]:.4f}, "
                   f"{incumbent['valid_ci'][1]:.4f}] — not distinguishable from "
                   f"noise. Keeping the shipped model.")
    print(verdict)

    # The test set is touched exactly once, by whichever model we would ship.
    shipped = winner if (decisive or winner["name"] == incumbent["name"]) else incumbent
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
        "variants": [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in results],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten to {args.out}")

    if args.save and shipped["name"] != incumbent["name"]:
        save_artifacts(SERVICE_DIR / "binary_truth_mlp.pkl", shipped["_model"],
                       shipped["_vectorizer"], shipped["_train_max_values"])
        print("saved the winner over binary_truth_mlp.pkl")
    elif args.save:
        print("nothing to save — the shipped model won.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
