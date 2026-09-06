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
    metrics = binary_metrics(labels_to_binary(test_df["label"]), probabilities, model.best_threshold)
    return {
        "evaluation": "claim-only production-equivalent LIAR test",
        "threshold": round(float(model.best_threshold), 4),
        "warning": "This is a dated US-political dataset; it is not a general-news accuracy claim.",
        **metrics,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_claim_only_model(), indent=2))
