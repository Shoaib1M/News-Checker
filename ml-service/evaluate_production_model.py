"""Evaluate the legacy MLP under the same inputs the live API receives.

The original LIAR evaluation includes speaker metadata and historical truth
counts.  The public endpoint receives only a statement, so this script reports
the claim-only metric separately and prevents accidental production overclaims.

Run from ml-service:
    python evaluate_production_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import (
    COLUMNS,
    labels_to_binary,
    load_artifacts,
    make_prediction_features_batch,
)


def bootstrap_interval(y_true, predictions, resamples=2000, seed=0):
    """95% confidence interval for accuracy, by resampling the test set.

    WHY THIS EXISTS:
    A single 1267-row split gives one number and no sense of how much of it is
    luck. Without an interval there is no way to tell a real improvement from
    noise, and the temptation is to chase the third decimal place of a figure
    whose second decimal is not stable. The gap that matters here — 61.9%
    against a 56.4% baseline — should be reported as an interval so a reader
    can see whether it clears the baseline at all.
    """
    rng = np.random.default_rng(seed)
    correct = (predictions == y_true).astype(float)
    n = len(correct)
    means = np.array([
        correct[rng.integers(0, n, n)].mean() for _ in range(resamples)
    ])
    return round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


def calibration_report(y_true, probabilities, bins=10):
    """How closely the predicted probability matches the observed frequency.

    WHY THIS MATTERS MORE THAN ACCURACY HERE:
    This score is shown to users and consumed downstream as a prior. A model
    that is 62% accurate but says "0.9" when it means "0.6" is worse than a
    less accurate model that knows what it does not know. Expected calibration
    error is the weighted average gap between the two, so 0.0 is perfect and
    anything above ~0.1 means the number should not be displayed as a
    confidence.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    total_gap = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities >= low) & (probabilities < high if high < 1.0
                                           else probabilities <= 1.0)
        count = int(in_bin.sum())
        if not count:
            continue
        predicted = float(probabilities[in_bin].mean())
        observed = float(y_true[in_bin].mean())
        rows.append({
            "range": f"{low:.1f}-{high:.1f}",
            "n": count,
            "mean_predicted": round(predicted, 4),
            "observed_frequency": round(observed, 4),
            "gap": round(predicted - observed, 4),
        })
        total_gap += count * abs(predicted - observed)
    return {
        "expected_calibration_error": round(total_gap / len(y_true), 4),
        "bins": rows,
    }


def binary_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": int(len(y_true)),
        "accuracy": round(float((predictions == y_true).mean()), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier_score": round(float(np.mean((probabilities - y_true) ** 2)), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate_claim_only_model():
    model_path = SERVICE_DIR / "binary_truth_mlp.pkl"
    if not model_path.exists():
        model_path = SERVICE_DIR / "saved_models" / "binary_truth_mlp.pkl"
    model, vectorizer, train_max_values = load_artifacts(model_path)

    test_df = pd.read_csv(SERVICE_DIR / "data" / "test.tsv", sep="\t", names=COLUMNS)
    # Built by the same function main.py calls, so this cannot drift from what
    # a live request computes. It used to transform the raw statement while
    # main.py went through build_text_input(), which prepends column-name
    # tokens and creates boundary bigrams -- a difference the old comment here
    # claimed did not exist. Measured, it was worth 0.5 points of accuracy
    # (0.6235 reported against 0.6188 actually served).
    production_features = make_prediction_features_batch(
        vectorizer, train_max_values, test_df["statement"].fillna("").astype(str)
    )
    probabilities = model.predict_proba(production_features)
    y_true = labels_to_binary(test_df["label"])
    metrics = binary_metrics(y_true, probabilities, model.best_threshold)
    predictions = (probabilities >= model.best_threshold).astype(int)
    low, high = bootstrap_interval(y_true, predictions)

    # The score a model gets for always predicting the larger class. An
    # accuracy figure without it is unreadable.
    baseline = float(max(y_true.mean(), 1 - y_true.mean()))

    return {
        "evaluation": "claim-only production-equivalent LIAR test",
        "accuracy_95_ci": [low, high],
        "majority_class_baseline": round(baseline, 4),
        "beats_baseline": bool(low > baseline),
        "calibration": calibration_report(y_true, probabilities),
        "threshold": round(float(model.best_threshold), 4),
        "warning": "This is a dated US-political dataset; it is not a general-news accuracy claim.",
        **metrics,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_claim_only_model(), indent=2))
