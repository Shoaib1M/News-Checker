"""
FILE PURPOSE:
Train and evaluate the LIAR model in its **standard benchmark setup** — the
statement plus the speaker context the dataset ships — and merge the result
into `evaluation_results.json` beside the claim-only model that actually
serves requests.

    python train_speaker_context_model.py

WHY THIS EXISTS:
Two different questions get the same name, and conflating them is how this
project ended up publishing a number it could not defend.

  - "What can the API do with a claim someone pasted?"  -> claim-only, 61.88%
  - "How does the model do on the LIAR benchmark?"      -> with speaker context

Both are legitimate. Only the first describes the product. Reporting the second
as if it were the first is the mistake, and the README carried a warning about
a "72.38%" figure for exactly that reason.

THE 72.38% WAS ALSO LEAKED, which is the part nobody had named. LIAR's credit
history includes the current statement's own label — see
`build_history_features` and `tests/test_history_leakage.py`. This script
passes `deleak_labels`, so the history means "the speaker's OTHER statements".
That costs accuracy and buys a number that survives being asked about.

NOT SHIPPED. The API never receives speaker metadata, so this model cannot
serve traffic; it exists to be reported honestly next to the one that does. No
pickle is committed — rerun this script to reproduce the figures.
"""

from __future__ import annotations

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
    BinaryTruthMLP,
    build_history_features,
    build_text_input,
    find_best_threshold,
    labels_to_binary,
    normalize_rows,
)
from tfidf import TFIDFVectorizer  # noqa: E402

RESULTS = SERVICE_DIR.parent / "client" / "public" / "evaluation_results.json"


def bootstrap_ci(y_true, predictions, resamples=2000, seed=0):
    rng = np.random.default_rng(seed)
    correct = (predictions == y_true).astype(float)
    n = len(correct)
    means = [correct[rng.integers(0, n, n)].mean() for _ in range(resamples)]
    return round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


def main() -> int:
    frames = {name: pd.read_csv(SERVICE_DIR / "data" / f"{name}.tsv", sep="\t",
                                names=COLUMNS)
              for name in ("train", "valid", "test")}
    labels = {name: labels_to_binary(f["label"]) for name, f in frames.items()}

    vectorizer = TFIDFVectorizer()
    vectorizer.build_vocab(build_text_input(frames["train"]))

    def features(name, train_max_values=None):
        frame = frames[name]
        text = normalize_rows(vectorizer.transform(build_text_input(frame)))
        # deleak_labels is the whole point: without it the history contains the
        # answer and every number below is inflated.
        history, train_max_values = build_history_features(
            frame, train_max_values, deleak_labels=frame["label"])
        return np.hstack([text, history]), train_max_values

    X_train, train_max_values = features("train")
    X_valid, _ = features("valid", train_max_values)
    X_test, _ = features("test", train_max_values)
    print(f"features: {X_train.shape[1]} "
          f"(statement + subject/speaker/job/state/party/context + de-leaked history)")

    started = time.time()
    model = BinaryTruthMLP(input_size=X_train.shape[1], hidden_size=64,
                           learning_rate=0.05, epochs=70, batch_size=128, seed=42)
    model.fit(X_train, labels["train"], X_valid, labels["valid"], quiet=True)

    # Threshold on validation, never on test.
    threshold, valid_accuracy = find_best_threshold(
        model.predict_proba(X_valid), labels["valid"])
    scores = model.predict_proba(X_test)
    predictions = (scores >= threshold).astype(int)
    y_test = labels["test"]

    accuracy = float((predictions == y_test).mean())
    low, high = bootstrap_ci(y_test, predictions)
    tp = int(((predictions == 1) & (y_test == 1)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())
    tn = int(((predictions == 0) & (y_test == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"\ntrained in {time.time() - started:.0f}s")
    print(f"  valid accuracy {valid_accuracy:.4f} | threshold {threshold:.2f}")
    print(f"  TEST accuracy  {accuracy:.4f}  95% CI [{low:.4f}, {high:.4f}]")
    print(f"  precision {precision:.4f} | recall {recall:.4f} | F1 {f1:.4f}")

    result = {
        "name": "Binary Truth MLP + speaker context",
        "type": "binary",
        "accuracy": round(accuracy, 4),
        "accuracy_95_ci": [low, high],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "labels": ["Fake-ish", "True-ish"],
        "architecture": "1 hidden layer (64 neurons, ReLU) → sigmoid",
        "input_features": (
            "statement + subject/speaker/job/state/party/context, plus credit "
            "history with the current statement removed"
        ),
        "training": "70 epochs, mini-batch SGD (batch=128), threshold on validation",
        "classes": "2 (binary)",
        "is_production": False,
        "note": (
            "The standard LIAR benchmark setup. NOT what the API can do: a "
            "pasted claim carries no speaker metadata. Credit history is "
            "de-leaked — LIAR's counts include the current statement's own "
            "label, which is where the commonly-quoted 72% figures come from."
        ),
    }

    if RESULTS.exists():
        payload = json.loads(RESULTS.read_text())
        payload.setdefault("models", {})["binary_mlp_speaker_context"] = result
        RESULTS.write_text(json.dumps(payload, indent=2))
        print(f"\nmerged into {RESULTS}")
    else:
        print(f"\n{RESULTS} not found — run evaluate_models.py first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
