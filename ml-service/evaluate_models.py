"""
Evaluate all three models on the LIAR test set and write
evaluation_results.json into client/public/ for the frontend to display.

Models evaluated:
  1. Logistic Regression (binary) — classifier.py
  2. MLP 6-class              — mlp_classifier.py
  3. Binary Truth Model        — binary_truth_mlp.py (the one that ships)

Run:
    cd ml-service
    python evaluate_models.py

WHAT THE COMPARISON IS FOR, AND WHAT IT WAS ACTUALLY SHOWING:
The page exists to answer "is the extra machinery earning its keep?". For that
to mean anything the comparison models have to be trained. Both of them were
not. Measured on this very test set:

  Logistic Regression   scored 56.35% — the majority-class rate to four
                        decimals — because 100 full-batch steps at lr 0.1 left
                        every one of its 1,267 scores inside [0.5543, 0.5597].
                        It predicted "true-ish" for every row.
  MLP 6-class           scored 20.84% against a 20.92% majority baseline: worse
                        than always answering "half-true", and it too predicted
                        a single class for every row.

So the production model was being compared against two models that had not
learned anything, which flatters it. Both now use settings that converge; the
gap that remains is a real one.
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
    LinearTruthModel,
    HISTORY_COLUMNS,
    SHIPPED_MODEL,
    SHIPPED_VECTORIZER,
    build_feature_frame,
    build_text_input,
    labels_to_binary,
    load_artifacts,
    load_split,
    make_prediction_features_batch,
    make_training_features,
    predict_proba_texts,
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
    """ROC curve points for plotting, sampled on a threshold grid."""
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


def compute_auc(y_true, scores):
    """Exact ROC-AUC by rank, with ties averaged.

    WHY NOT THE TRAPEZOID OVER compute_roc_curve():
    That integrated a curve sampled on 200 fixed thresholds in [0, 1] and then
    ROUNDED each coordinate to four decimals before summing. When a model's
    scores are bunched — and the old under-trained baselines put every score
    inside a 0.006-wide band — nearly all 200 samples land on the same corner
    of the curve and the integral is of whatever the rounding left behind.
    The rank form has no grid, no rounding and no free parameters: it is the
    probability that a randomly chosen true-ish claim outranks a randomly
    chosen fake-ish one, computed exactly.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    index = 0
    while index < len(sorted_scores):
        end = index
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1

    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(float(auc), 4)


def describe_architecture(model):
    """Read the description off the object rather than hard-coding it.

    The three "Model Details" strings on the frontend were literals, so they
    kept saying "1 hidden layer (64 neurons, ReLU)" and "70 epochs" no matter
    what was actually in the pickle. A description that cannot go stale is
    worth more than a prettier one.
    """
    if isinstance(model, LinearTruthModel):
        return "Linear (no hidden layer) → sigmoid, L2-regularised"
    return f"1 hidden layer ({model.hidden_size} neurons, ReLU) → sigmoid"


def describe_features(vectorizer):
    parts = []
    if vectorizer.ngram_range:
        low, high = vectorizer.ngram_range
        parts.append(f"word {low}-{high}grams")
    if vectorizer.char_ngram_range:
        low, high = vectorizer.char_ngram_range
        parts.append(f"character {low}-{high}grams")
    weighting = "sublinear TF" if vectorizer.sublinear_tf else "raw TF"
    return (f"TF-IDF ({' + '.join(parts)}, {weighting}, min_df={vectorizer.min_df}, "
            f"{vectorizer.vocab_size:,} features) of the statement only")


def describe_training(model):
    if isinstance(model, LinearTruthModel):
        return (f"{model.epochs} epochs full-batch gradient descent, L2={model.l2:g}, "
                f"threshold tuned on the validation set")
    return (f"{model.epochs} epochs, mini-batch SGD (batch={model.batch_size}), "
            f"threshold tuned on the validation set")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading LIAR dataset splits...")
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df = load_split("test")

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

    # No learning_rate/epochs override: this call site used to pin lr=0.1 for
    # 100 steps, which never moved the decision boundary off the majority
    # class. classifier.py now carries settings that converge; see the class
    # docstring there.
    lr_model = LogisticRegression()
    lr_model.fit(X_train_lr, y_train_lr)

    lr_scores = lr_model.predict_proba(X_test_lr)
    lr_preds = lr_model.predict(X_test_lr)
    lr_acc = float(np.mean(lr_preds == y_test_lr))
    lr_precision, lr_recall, lr_f1 = precision_recall_f1_binary(y_test_lr, lr_preds)
    lr_cm = confusion_matrix_binary(y_test_lr, lr_preds)
    lr_roc = compute_roc_curve(y_test_lr, lr_scores)
    lr_auc = compute_auc(y_test_lr, lr_scores)

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
        "input_features": describe_features(lr_vectorizer),
        "training": (f"{lr_model.epochs} epochs, full-batch gradient descent "
                     f"with momentum, L2={lr_model.l2:g}"),
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

    # No learning_rate/epochs override, for the same reason as the logistic
    # regression above: lr=0.03 for 30 epochs scored 20.84% here, below the
    # 20.92% you get from always answering "half-true". mlp_classifier.py now
    # carries settings that train.
    mlp6 = MLPClassifier(
        input_size=X_train_mlp.shape[1],
        hidden_size=64,
        output_size=len(LABELS_6),
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
        "input_features": describe_features(mlp_vectorizer),
        "training": (f"{mlp6.epochs} epochs, mini-batch SGD (batch={mlp6.batch_size}), "
                     f"lr {mlp6.lr} chosen on the validation split"),
        "classes": "6 (fine-grained)",
    }

    # ── 3. Production claim model ─────────────────────────────────────────
    print("\n=== Evaluating the production claim model ===")

    if MODEL_FILE.exists():
        print("  Loading pre-trained model...")
        bt_model, bt_vectorizer, train_max_values = load_artifacts(MODEL_FILE)
    else:
        print("  No saved model found — training from scratch...")
        # Statement-only, history zeroed -- identical to binary_truth_mlp.main().
        # This branch used to train on speaker metadata and real history counts,
        # producing a DIFFERENT model from the one that ships, so the fallback
        # silently changed what the comparison page was reporting.
        train_text = train_df["statement"].fillna("").astype(str)
        valid_text = valid_df["statement"].fillna("").astype(str)
        bt_vectorizer = TFIDFVectorizer(**SHIPPED_VECTORIZER)
        bt_vectorizer.build_vocab(
            build_text_input(build_feature_frame(train_text)))

        train_max_values = np.ones((1, len(HISTORY_COLUMNS)))
        X_train_bt = make_training_features(bt_vectorizer, train_text)
        X_valid_bt = make_training_features(bt_vectorizer, valid_text)
        y_train_bt = labels_to_binary(train_df["label"])
        y_valid_bt = labels_to_binary(valid_df["label"])

        bt_model = LinearTruthModel(input_size=X_train_bt.shape[1], **SHIPPED_MODEL)
        bt_model.fit(X_train_bt, y_train_bt, X_valid_bt, y_valid_bt)

    # Evaluate on the test set through the SAME feature construction a live
    # request uses. This block previously built its own features from
    # build_text_input(test_df) -- statement plus speaker, job, state, party and
    # context -- with real non-zero history counts, none of which the shipped
    # model was trained on. It scored 56.9% for a model that gets 61.9% on the
    # inputs it actually receives.
    # predict_proba_texts() calls make_prediction_features_batch() in chunks --
    # same construction, bounded memory. The dense serving form of 1,267 rows
    # against 62,257 features would be 631 MB in one allocation.
    y_test_bt = labels_to_binary(test_df["label"])
    bt_scores = predict_proba_texts(
        bt_model, bt_vectorizer, train_max_values,
        test_df["statement"].fillna("").astype(str),
    )
    bt_preds = (bt_scores >= bt_model.best_threshold).astype(int)
    bt_acc = float(np.mean(bt_preds == y_test_bt))
    bt_precision, bt_recall, bt_f1 = precision_recall_f1_binary(y_test_bt, bt_preds)
    bt_cm = confusion_matrix_binary(y_test_bt, bt_preds)
    bt_roc = compute_roc_curve(y_test_bt, bt_scores)
    bt_auc = compute_auc(y_test_bt, bt_scores)

    print(f"  Accuracy:  {bt_acc * 100:.2f}%")
    print(f"  Precision: {bt_precision:.4f}")
    print(f"  Recall:    {bt_recall:.4f}")
    print(f"  F1:        {bt_f1:.4f}")
    print(f"  AUC:       {bt_auc:.4f}")
    print(f"  Threshold: {bt_model.best_threshold:.2f}")

    bt_result = {
        "name": "Binary Truth Model",
        "type": "binary",
        "accuracy": round(bt_acc, 4),
        "precision": round(bt_precision, 4),
        "recall": round(bt_recall, 4),
        "f1": round(bt_f1, 4),
        "confusion_matrix": bt_cm,
        "labels": BINARY_LABELS,
        "roc_curve": bt_roc,
        "auc": bt_auc,
        "architecture": describe_architecture(bt_model),
        "input_features": describe_features(bt_vectorizer),
        "training": describe_training(bt_model),
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
