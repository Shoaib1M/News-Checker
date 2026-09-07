"""
FILE PURPOSE:
Choose the claim model's configuration, honestly, and write the winner.

    python train_experiments.py                 # sweep, report, recommend
    python train_experiments.py --save          # also write the winner to disk
    python train_experiments.py --only "linear, sublinear TF, word only" --save

WHY THIS EXISTS:
The shipped model's hyperparameters were chosen once and never revisited, and
the training loop had no regularisation and no early stopping on 26,626 input
dimensions over ~10k rows — the textbook setup for overfitting. Whether that
costs anything is an empirical question, and this answers it.

────────────────────────────────────────────────────────────────────────────
THE RULE THIS SCRIPT ENFORCES: **SELECTION NEVER SEES THE TEST SET.**
────────────────────────────────────────────────────────────────────────────
Running a dozen variants and shipping whichever scored best on test is
overfitting the test set with extra steps — the reported number then describes
the sweep, not the model, and it will not survive contact with new data.
`evaluate_production_model.py` reports the shipped model's test metrics; this
script's job is to decide *which* model that should be.

WHY IT CROSS-VALIDATES INSTEAD OF USING THE VALIDATION SPLIT:
The first version of this script selected on `valid.tsv` — one split, 1,284
rows. That is enough to rank two very different models and nowhere near enough
to separate two good ones: the 95% interval on a validation accuracy there is
about ±2.6 points, and the differences being chased are one to two. The cost
of that was measurable. The word+char configuration beat the incumbent by
**+3.04 points on validation, CI [+0.70, +5.37]** — and by **+1.34 points on
the held-out test set, CI [-0.87, +3.55]**. Most of the validation margin was
the split, not the model.

Five-fold cross-validation over train+valid scores every configuration on
11,524 rows instead of 1,284, which cuts the standard error of the comparison
by about half and, more importantly, averages over *which* rows land in the
evaluation set. Folds are shared across configurations, so every variant is
scored on exactly the same partitions and the comparison is paired.

The decision threshold comes from the pooled out-of-fold predictions, so it is
also never fitted on data the model was trained on.
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
    LinearTruthModel,
    build_feature_frame,
    build_text_input,
    find_best_threshold,
    labels_to_binary,
    load_split,
    make_training_features,
    save_artifacts,
)
from tfidf import TFIDFVectorizer  # noqa: E402

# (name, vectorizer kwargs, model class, model kwargs)
#
# Deliberately small. Each variant tests one idea against the incumbent, so a
# difference can be attributed; a grid of everything-against-everything would
# mostly measure noise, and cross-validation makes each variant five trainings
# rather than one.
BASE_VECTORIZER = {"ngram_range": (1, 2), "min_df": 2}
MODERN = {"sublinear_tf": True, "smooth_idf": True}

VARIANTS = [
    # The incumbent: no regularisation, no early stopping, raw TF, word only.
    ("mlp baseline (was shipped)", BASE_VECTORIZER, BinaryTruthMLP,
     {"hidden_size": 64, "learning_rate": 0.05, "epochs": 70, "batch_size": 128}),
    # Is the hidden layer the problem, or was it just the learning rate? An
    # earlier single-split sweep found lr 0.5 with early stopping to be the
    # best MLP configuration, so that is the one the hidden layer gets to
    # defend itself with.
    ("mlp, lr 0.5 + early stop", BASE_VECTORIZER, BinaryTruthMLP,
     {"hidden_size": 64, "learning_rate": 0.5, "epochs": 70, "batch_size": 128,
      "weight_decay": 1e-3, "early_stopping_patience": 5}),
    # Drop the hidden layer, keep everything else. Isolates depth.
    ("linear, raw TF, word only", BASE_VECTORIZER, LinearTruthModel, {"l2": 1e-4}),
    # Then the representation changes, one at a time.
    ("linear, sublinear TF, word only", {**BASE_VECTORIZER, **MODERN},
     LinearTruthModel, {"l2": 1e-4}),
    ("linear, word + char 3-5", {**BASE_VECTORIZER, **MODERN, "char_ngram_range": (3, 5)},
     LinearTruthModel, {"l2": 1e-4}),
    ("linear, word + char 3-6", {**BASE_VECTORIZER, **MODERN, "char_ngram_range": (3, 6)},
     LinearTruthModel, {"l2": 1e-4}),
    ("linear, char 3-5 only", {"ngram_range": None, "min_df": 2, **MODERN,
                               "char_ngram_range": (3, 5)}, LinearTruthModel, {"l2": 1e-4}),
    # Regularisation strength, on the best representation so far.
    ("linear, word+char, L2 1e-5", {**BASE_VECTORIZER, **MODERN, "char_ngram_range": (3, 5)},
     LinearTruthModel, {"l2": 1e-5}),
    ("linear, word+char, L2 1e-3", {**BASE_VECTORIZER, **MODERN, "char_ngram_range": (3, 5)},
     LinearTruthModel, {"l2": 1e-3}),
    ("linear, word+char, L2 1e-2", {**BASE_VECTORIZER, **MODERN, "char_ngram_range": (3, 5)},
     LinearTruthModel, {"l2": 1e-2}),
]

N_FOLDS = 5
FOLD_SEED = 20240


def bootstrap_interval(y_true, predictions, resamples=2000, seed=0):
    rng = np.random.default_rng(seed)
    correct = (np.asarray(predictions) == np.asarray(y_true)).astype(float)
    n = len(correct)
    means = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_difference(y_true, baseline_pred, variant_pred, resamples=2000, seed=1):
    """95% interval on accuracy(variant) - accuracy(baseline), resampling rows.

    Paired, because both models were scored on the same rows: the shared
    difficulty of those rows cancels, and what is left is the difference
    between the models. An unpaired comparison of two overlapping intervals
    would call almost everything here inconclusive.
    """
    rng = np.random.default_rng(seed)
    base = (np.asarray(baseline_pred) == np.asarray(y_true)).astype(float)
    variant = (np.asarray(variant_pred) == np.asarray(y_true)).astype(float)
    n = len(base)
    diffs = np.array([
        variant[i].mean() - base[i].mean()
        for i in (rng.integers(0, n, n) for _ in range(resamples))
    ])
    return float(np.mean(variant - base)), float(np.percentile(diffs, 2.5)), \
        float(np.percentile(diffs, 97.5))


def roc_auc(y_true, scores):
    """Exact rank AUC, ties averaged."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ordered = scores[order]
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_vectorizer(kwargs, statements):
    """Vocabulary from the TRAINING FOLD only.

    Fitting it on anything else leaks the evaluation rows into the feature
    space — the IDF weights alone would encode which words the held-out rows
    contain — and inflates every number that follows.
    """
    vectorizer = TFIDFVectorizer(**kwargs)
    vectorizer.build_vocab(build_text_input(build_feature_frame(statements)))
    return vectorizer


def out_of_fold_scores(vectorizer_kwargs, model_class, model_kwargs, statements, labels, folds):
    """Train on four folds, score the fifth, five times. Every row gets a score
    from a model that never saw it."""
    scores = np.zeros(len(labels))
    for fold in range(N_FOLDS):
        holdout = folds == fold
        train_text = [statements[i] for i in np.flatnonzero(~holdout)]
        holdout_text = [statements[i] for i in np.flatnonzero(holdout)]
        vectorizer = build_vectorizer(vectorizer_kwargs, train_text)
        X_train = make_training_features(vectorizer, train_text)
        X_holdout = make_training_features(vectorizer, holdout_text)
        model = model_class(input_size=X_train.shape[1], **model_kwargs)
        if model_class is BinaryTruthMLP:
            # The legacy model cannot read a sparse matrix; it is being
            # measured, not shipped, so the dense conversion is acceptable.
            model.fit(X_train.to_dense(), labels[~holdout], quiet=True)
            scores[holdout] = model.predict_proba(X_holdout.to_dense())
        else:
            model.fit(X_train, labels[~holdout], quiet=True)
            scores[holdout] = model.predict_proba(X_holdout)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="write the winning model over binary_truth_mlp.pkl")
    parser.add_argument("--out", default="experiment_results.json")
    # Not comma-separated: variant names contain commas ("linear, word + char
    # 3-5"), and splitting on them turned one name into three unknown ones.
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="run only this variant; repeat the flag for several")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    wanted = {name.strip() for name in args.only if name.strip()}
    if wanted:
        unknown = wanted - {name for name, *_ in VARIANTS}
        if unknown:
            raise SystemExit(f"unknown variant(s): {sorted(unknown)}\n"
                             f"available: {[n for n, *_ in VARIANTS]}")

    train_df = load_split("train")
    valid_df = load_split("valid")

    # train + valid is the SELECTION pool. test.tsv is not read by this script.
    pool = pd.concat([train_df, valid_df], ignore_index=True)
    statements = pool["statement"].fillna("").astype(str).tolist()
    labels = labels_to_binary(pool["label"])

    rng = np.random.default_rng(FOLD_SEED)
    folds = rng.permutation(np.arange(len(labels)) % args.folds)
    baseline = float(max(labels.mean(), 1 - labels.mean()))

    print(f"\nselection pool: {len(labels)} rows (train + valid), {args.folds}-fold CV")
    print(f"majority-class baseline: {baseline:.4f}")
    print("test.tsv is NOT read by this script.\n")
    print(f"{'variant':<32}{'CV acc':>8}{'95% CI':>18}{'AUC':>8}{'Brier':>8}{'thr':>6}{'secs':>7}")
    print("-" * 89)

    results = []
    incumbent_scores = None
    for name, vectorizer_kwargs, model_class, model_kwargs in VARIANTS:
        if wanted and name not in wanted:
            continue
        started = time.time()
        scores = out_of_fold_scores(vectorizer_kwargs, model_class, model_kwargs,
                                    statements, labels, folds)
        threshold, accuracy = find_best_threshold(scores, labels)
        low, high = bootstrap_interval(labels, (scores >= threshold).astype(int))
        elapsed = time.time() - started
        entry = {
            "name": name, "cv_accuracy": round(float(accuracy), 4),
            "cv_ci": [round(low, 4), round(high, 4)],
            "cv_auc": round(roc_auc(labels, scores), 4),
            "cv_brier": round(float(np.mean((scores - labels) ** 2)), 4),
            "threshold": round(float(threshold), 3),
            "seconds": round(elapsed, 1),
            "model": model_class.__name__,
            "vectorizer": {k: list(v) if isinstance(v, tuple) else v
                           for k, v in vectorizer_kwargs.items()},
            "model_kwargs": model_kwargs,
        }
        if incumbent_scores is None:
            incumbent_scores = scores
        else:
            delta, delta_low, delta_high = paired_difference(
                labels, (incumbent_scores >= 0.5).astype(int),
                (scores >= threshold).astype(int))
            entry["vs_incumbent"] = [round(delta, 4), round(delta_low, 4), round(delta_high, 4)]
        results.append(entry)
        print(f"{name:<32}{accuracy:>8.4f}  [{low:.4f}, {high:.4f}]{entry['cv_auc']:>8.4f}"
              f"{entry['cv_brier']:>8.4f}{threshold:>6.2f}{elapsed:>7.0f}", flush=True)
        Path(args.out).write_text(json.dumps(
            {"pool_rows": int(len(labels)), "folds": args.folds,
             "majority_baseline": round(baseline, 4), "variants": results}, indent=2))

    if not results:
        raise SystemExit("no variants ran")

    # THE SELECTION CRITERION, fixed before the sweep ran: highest
    # cross-validated accuracy at the out-of-fold-tuned threshold, tie-broken
    # (within 0.002) by AUC. Writing it down in advance is what stops the
    # criterion from being chosen after the fact to favour a preferred answer.
    best = max(results, key=lambda r: (round(r["cv_accuracy"], 3), r["cv_auc"]))
    print(f"\nWINNER ON CROSS-VALIDATION: {best['name']}  "
          f"({best['cv_accuracy']:.4f}, AUC {best['cv_auc']:.4f})")
    for row in results:
        if "vs_incumbent" in row and row["name"] == best["name"]:
            delta, low, high = row["vs_incumbent"]
            verdict = "clears zero" if low > 0 else "NOT distinguishable from the incumbent"
            print(f"  vs the incumbent: {delta*100:+.2f} pts, 95% CI "
                  f"[{low*100:+.2f}, {high*100:+.2f}] — {verdict}")

    payload = {"pool_rows": int(len(labels)), "folds": args.folds,
               "majority_baseline": round(baseline, 4), "winner": best["name"],
               "variants": results}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten to {args.out}")

    if args.save:
        # Retrain the winner on the WHOLE selection pool. The out-of-fold
        # threshold carries over: it was fitted on predictions from models that
        # had not seen the rows they scored, so it is not tuned on this fit.
        settings = next(v for v in VARIANTS if v[0] == best["name"])
        _, vectorizer_kwargs, model_class, model_kwargs = settings
        vectorizer = build_vectorizer(vectorizer_kwargs, statements)
        X = make_training_features(vectorizer, statements)
        model = model_class(input_size=X.shape[1], **model_kwargs)
        model.fit(X if model_class is not BinaryTruthMLP else X.to_dense(),
                  labels, quiet=True)
        model.best_threshold = best["threshold"]
        save_artifacts(SERVICE_DIR / "binary_truth_mlp.pkl", model, vectorizer,
                       np.ones((1, len(HISTORY_COLUMNS))))
        print(f"saved '{best['name']}' over binary_truth_mlp.pkl "
              f"({X.shape[1]} features, threshold {best['threshold']:.2f})")
        print("Now run evaluate_production_model.py for its held-out test number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
