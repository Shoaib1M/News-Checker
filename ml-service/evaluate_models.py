"""
Evaluate all three models on the LIAR test set and write
evaluation_results.json into client/public/ for the frontend to display.

Models evaluated:
  1. Logistic Regression (binary) — classifier.py
  2. MLP 6-class              — mlp_classifier.py
  3. Binary Truth MLP          — binary_truth_mlp.py

Run:
    cd ml-service
    python evaluate_models.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sibling modules importable
# ---------------------------------------------------------------------------
SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from tfidf import TFIDFVectorizer
from classifier import LogisticRegression
from mlp_classifier import MLPClassifier, LABELS as LABELS_6
from binary_truth_mlp import (
    BinaryTruthMLP,
    build_text_input,
    build_history_features,
    labels_to_binary,
    normalize_rows,
    load_artifacts,
    MODEL_FILE,
    COLUMNS,
    FAKEISH_LABELS,
)


DATA_DIR = SERVICE_DIR / "data"
OUTPUT_PATH = SERVICE_DIR.parent / "client" / "public" / "evaluation_results.json"

BINARY_LABELS = ["Fake-ish", "True-ish"]


# ── helpers ──────────────────────────────────────────────────────────────────

def confusion_matrix_binary(y_true, y_pred):
    """Return 2×2 confusion matrix [[TN, FP], [FN, TP]]."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return [[tn, fp], [fn, tp]]


def confusion_matrix_multi(y_true, y_pred, n_classes):
    """Return n×n confusion matrix."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix.tolist()


def precision_recall_f1_binary(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def precision_recall_f1_per_class(y_true, y_pred, n_classes):
    metrics = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = int(np.sum(y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.append({
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        })
    return metrics


def compute_roc_curve(y_true, scores, n_points=200):
    """Compute ROC curve data points."""
    thresholds = np.linspace(0, 1, n_points)
    points = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        points.append({"fpr": round(float(fpr), 4), "tpr": round(float(tpr), 4)})

    # Sort by fpr for clean plotting
    points.sort(key=lambda p: (p["fpr"], p["tpr"]))
    return points


def compute_auc(roc_points):
    """Trapezoidal AUC from ROC points."""
    fprs = [p["fpr"] for p in roc_points]
    tprs = [p["tpr"] for p in roc_points]
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
    return round(float(auc), 4)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading LIAR dataset splits...")
    train_df = pd.read_csv(DATA_DIR / "train.tsv", sep="\t", names=COLUMNS)
    valid_df = pd.read_csv(DATA_DIR / "valid.tsv", sep="\t", names=COLUMNS)
    test_df = pd.read_csv(DATA_DIR / "test.tsv", sep="\t", names=COLUMNS)

    dataset_info = {
        "name": "LIAR (Politifact)",
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "labels_6class": LABELS_6,
        "binary_mapping": "pants-fire / false / barely-true → Fake-ish  |  half-true / mostly-true / true → True-ish",
        "label_distribution": {
            label: int(np.sum(test_df["label"] == label)) for label in LABELS_6
        },
    }

    # ── 1. Logistic Regression (binary) ───────────────────────────────────
    print("\n=== Evaluating Logistic Regression ===")
    lr_vectorizer = TFIDFVectorizer()
    lr_vectorizer.build_vocab(train_df["statement"])

    X_train_lr = lr_vectorizer.transform(train_df["statement"])
    X_test_lr = lr_vectorizer.transform(test_df["statement"])
    X_train_lr = X_train_lr / (np.linalg.norm(X_train_lr, axis=1, keepdims=True) + 1e-9)
    X_test_lr = X_test_lr / (np.linalg.norm(X_test_lr, axis=1, keepdims=True) + 1e-9)

    fake_labels = {"pants-fire", "false", "barely-true"}
    y_train_lr = train_df["label"].apply(lambda x: 0 if x in fake_labels else 1).values
    y_test_lr = test_df["label"].apply(lambda x: 0 if x in fake_labels else 1).values

    lr_model = LogisticRegression(learning_rate=0.1, epochs=100)
    lr_model.fit(X_train_lr, y_train_lr)

    lr_scores = lr_model.predict_proba(X_test_lr)
    lr_preds = lr_model.predict(X_test_lr)
    lr_acc = float(np.mean(lr_preds == y_test_lr))
    lr_precision, lr_recall, lr_f1 = precision_recall_f1_binary(y_test_lr, lr_preds)
    lr_cm = confusion_matrix_binary(y_test_lr, lr_preds)
    lr_roc = compute_roc_curve(y_test_lr, lr_scores)
    lr_auc = compute_auc(lr_roc)

    print(f"  Accuracy:  {lr_acc * 100:.2f}%")
    print(f"  Precision: {lr_precision:.4f}")
    print(f"  Recall:    {lr_recall:.4f}")
    print(f"  F1:        {lr_f1:.4f}")
    print(f"  AUC:       {lr_auc:.4f}")

    lr_result = {
        "name": "Logistic Regression",
        "type": "binary",
        "accuracy": round(lr_acc, 4),
        "precision": round(lr_precision, 4),
        "recall": round(lr_recall, 4),
        "f1": round(lr_f1, 4),
        "confusion_matrix": lr_cm,
        "labels": BINARY_LABELS,
        "roc_curve": lr_roc,
        "auc": lr_auc,
        "architecture": "Single neuron (no hidden layer)",
        "input_features": "TF-IDF bigrams only",
        "training": "100 epochs, full-batch gradient descent",
        "classes": "2 (binary)",
        "threshold": 0.5,
    }

    # ── 2. MLP 6-class ────────────────────────────────────────────────────
    print("\n=== Evaluating MLP 6-Class ===")
    from mlp_classifier import (
        MLPClassifier,
        labels_to_numbers,
        accuracy as mlp_accuracy,
        normalize_rows as mlp_normalize,
    )

    mlp_vectorizer = TFIDFVectorizer()
    mlp_vectorizer.build_vocab(train_df["statement"])

    X_train_mlp = mlp_normalize(mlp_vectorizer.transform(train_df["statement"]))
    X_valid_mlp = mlp_normalize(mlp_vectorizer.transform(valid_df["statement"]))
    X_test_mlp = mlp_normalize(mlp_vectorizer.transform(test_df["statement"]))

    y_train_mlp = labels_to_numbers(train_df["label"])
    y_valid_mlp = labels_to_numbers(valid_df["label"])
    y_test_mlp = labels_to_numbers(test_df["label"])

    mlp6 = MLPClassifier(
        input_size=X_train_mlp.shape[1],
        hidden_size=64,
        output_size=len(LABELS_6),
        learning_rate=0.03,
        epochs=30,
        batch_size=128,
    )
    mlp6.fit(X_train_mlp, y_train_mlp, X_valid_mlp, y_valid_mlp)

    mlp6_preds = mlp6.predict(X_test_mlp)
    mlp6_acc = float(np.mean(mlp6_preds == y_test_mlp))
    mlp6_cm = confusion_matrix_multi(y_test_mlp, mlp6_preds, len(LABELS_6))
    mlp6_per_class = precision_recall_f1_per_class(y_test_mlp, mlp6_preds, len(LABELS_6))

    # Weighted-average metrics for 6-class
    total_support = sum(m["support"] for m in mlp6_per_class)
    mlp6_precision_w = sum(m["precision"] * m["support"] for m in mlp6_per_class) / total_support
    mlp6_recall_w = sum(m["recall"] * m["support"] for m in mlp6_per_class) / total_support
    mlp6_f1_w = sum(m["f1"] * m["support"] for m in mlp6_per_class) / total_support

    print(f"  Accuracy:           {mlp6_acc * 100:.2f}%")
    print(f"  Weighted Precision: {mlp6_precision_w:.4f}")
    print(f"  Weighted Recall:    {mlp6_recall_w:.4f}")
    print(f"  Weighted F1:        {mlp6_f1_w:.4f}")

    mlp6_result = {
        "name": "MLP 6-Class",
        "type": "multiclass",
        "accuracy": round(mlp6_acc, 4),
        "precision": round(mlp6_precision_w, 4),
        "recall": round(mlp6_recall_w, 4),
        "f1": round(mlp6_f1_w, 4),
        "per_class_metrics": mlp6_per_class,
        "confusion_matrix": mlp6_cm,
        "labels": LABELS_6,
        "architecture": "1 hidden layer (64 neurons, ReLU) → softmax",
        "input_features": "TF-IDF bigrams only",
        "training": "30 epochs, mini-batch SGD (batch=128)",
        "classes": "6 (fine-grained)",
    }

    # ── 3. Binary Truth MLP ───────────────────────────────────────────────
    print("\n=== Evaluating Binary Truth MLP ===")

    if MODEL_FILE.exists():
        print("  Loading pre-trained model...")
        bt_model, bt_vectorizer, train_max_values = load_artifacts(MODEL_FILE)
    else:
        print("  No saved model found — training from scratch...")
        bt_vectorizer = TFIDFVectorizer()
        train_text = build_text_input(train_df)
        bt_vectorizer.build_vocab(train_text)

        X_train_bt = normalize_rows(bt_vectorizer.transform(train_text))
        X_train_hist, train_max_values = build_history_features(train_df)
        X_train_bt = np.hstack([X_train_bt, X_train_hist])
        y_train_bt = labels_to_binary(train_df["label"])

        valid_text = build_text_input(valid_df)
        X_valid_bt = normalize_rows(bt_vectorizer.transform(valid_text))
        X_valid_hist, _ = build_history_features(valid_df, train_max_values)
        X_valid_bt = np.hstack([X_valid_bt, X_valid_hist])
        y_valid_bt = labels_to_binary(valid_df["label"])

        bt_model = BinaryTruthMLP(
            input_size=X_train_bt.shape[1],
            hidden_size=64,
            learning_rate=0.05,
            epochs=70,
            batch_size=128,
        )
        bt_model.fit(X_train_bt, y_train_bt, X_valid_bt, y_valid_bt)

    # Evaluate on test set
    test_text = build_text_input(test_df)
    X_test_bt = normalize_rows(bt_vectorizer.transform(test_text))
    X_test_hist, _ = build_history_features(test_df, train_max_values)
    X_test_bt = np.hstack([X_test_bt, X_test_hist])
    y_test_bt = labels_to_binary(test_df["label"])

    bt_scores = bt_model.predict_proba(X_test_bt)
    bt_preds = bt_model.predict(X_test_bt)
    bt_acc = float(np.mean(bt_preds == y_test_bt))
    bt_precision, bt_recall, bt_f1 = precision_recall_f1_binary(y_test_bt, bt_preds)
    bt_cm = confusion_matrix_binary(y_test_bt, bt_preds)
    bt_roc = compute_roc_curve(y_test_bt, bt_scores)
    bt_auc = compute_auc(bt_roc)

    print(f"  Accuracy:  {bt_acc * 100:.2f}%")
    print(f"  Precision: {bt_precision:.4f}")
    print(f"  Recall:    {bt_recall:.4f}")
    print(f"  F1:        {bt_f1:.4f}")
    print(f"  AUC:       {bt_auc:.4f}")
    print(f"  Threshold: {bt_model.best_threshold:.2f}")

    bt_result = {
        "name": "Binary Truth MLP",
        "type": "binary",
        "accuracy": round(bt_acc, 4),
        "precision": round(bt_precision, 4),
        "recall": round(bt_recall, 4),
        "f1": round(bt_f1, 4),
        "confusion_matrix": bt_cm,
        "labels": BINARY_LABELS,
        "roc_curve": bt_roc,
        "auc": bt_auc,
        "architecture": "1 hidden layer (64 neurons, ReLU) → sigmoid",
        "input_features": "TF-IDF bigrams + speaker metadata + history counts",
        "training": "70 epochs, mini-batch SGD (batch=128), threshold-tuned on validation set",
        "classes": "2 (binary)",
        "threshold": round(float(bt_model.best_threshold), 4),
        "is_production": True,
    }

    # ── Build final JSON ──────────────────────────────────────────────────
    output = {
        "dataset": dataset_info,
        "models": {
            "logistic_regression": lr_result,
            "mlp_6class": mlp6_result,
            "binary_mlp": bt_result,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Evaluation results written to: {OUTPUT_PATH}")
    print(f"   File size: {OUTPUT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
