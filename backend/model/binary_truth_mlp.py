from pathlib import Path

import numpy as np
import pandas as pd

from tfidf import TFIDFVectorizer


COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire",
    "context",
]

FAKEISH_LABELS = {"pants-fire", "false", "barely-true"}
TRUEISH_LABELS = {"half-true", "mostly-true", "true"}
TEXT_FEATURE_COLUMNS = ["statement", "subject", "speaker", "job", "state", "party", "context"]
HISTORY_COLUMNS = ["barely_true", "false", "half_true", "mostly_true", "pants_fire"]


class BinaryTruthMLP:
    def __init__(
        self,
        input_size,
        hidden_size=64,
        learning_rate=0.05,
        epochs=40,
        batch_size=128,
        patience=10,
        seed=42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.best_threshold = 0.5

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probability = self.sigmoid(z2)
        return z1, a1, probability

    def loss(self, predicted, actual):
        predicted = np.clip(predicted.flatten(), 1e-9, 1 - 1e-9)
        return -np.mean(
            actual * np.log(predicted) +
            (1 - actual) * np.log(1 - predicted)
        )

    def fit(self, X, y, X_valid=None, y_valid=None):
        n_samples = X.shape[0]
        best_weights = None
        best_valid_acc = -1
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices].reshape(-1, 1)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                current_batch_size = X_batch.shape[0]

                z1, a1, predicted = self.forward(X_batch)

                # For sigmoid + binary cross-entropy, this gradient simplifies nicely.
                dz2 = (predicted - y_batch) / current_batch_size
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self.relu_derivative(z1)
                dW1 = np.dot(X_batch.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            should_report = epoch == 1 or epoch % 5 == 0
            if should_report:
                train_pred = self.predict_proba(X)
                train_loss = self.loss(train_pred, y)
                train_acc = accuracy(train_pred, y, threshold=0.5)

                message = (
                    f"Epoch {epoch:03d} | "
                    f"loss: {train_loss:.4f} | "
                    f"train accuracy: {train_acc * 100:.2f}%"
                )

                if X_valid is not None and y_valid is not None:
                    valid_pred = self.predict_proba(X_valid)
                    valid_loss = self.loss(valid_pred, y_valid)
                    valid_threshold, valid_acc = find_best_threshold(valid_pred, y_valid)
                    message += (
                        f" | valid loss: {valid_loss:.4f} | "
                        f"valid accuracy: {valid_acc * 100:.2f}% | "
                        f"threshold: {valid_threshold:.2f}"
                    )

                print(message)

            if X_valid is not None and y_valid is not None and should_report:
                valid_pred = self.predict_proba(X_valid)
                valid_threshold, valid_acc = find_best_threshold(valid_pred, y_valid)

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_epoch = epoch
                    self.best_threshold = valid_threshold
                    best_weights = (
                        self.W1.copy(),
                        self.b1.copy(),
                        self.W2.copy(),
                        self.b2.copy(),
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 5

                if epochs_without_improvement >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"Best validation accuracy was {best_valid_acc * 100:.2f}% "
                        f"at epoch {best_epoch}."
                    )
                    break

        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights

    def predict_proba(self, X):
        _, _, probability = self.forward(X)
        return probability.flatten()

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        return (self.predict_proba(X) >= threshold).astype(int)


def labels_to_binary(labels):
    return labels.apply(lambda label: 1 if label in TRUEISH_LABELS else 0).values.astype(float)


def accuracy(predicted_scores, actual, threshold=0.5):
    predicted = predicted_scores >= threshold
    return np.mean(predicted == actual)


def find_best_threshold(predicted_scores, actual):
    best_threshold = 0.5
    best_acc = 0

    for threshold in np.arange(0.30, 0.71, 0.01):
        current_acc = accuracy(predicted_scores, actual, threshold)
        if current_acc > best_acc:
            best_acc = current_acc
            best_threshold = threshold

    return best_threshold, best_acc


def normalize_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def build_text_input(df):
    pieces = []
    for column in TEXT_FEATURE_COLUMNS:
        values = df[column].fillna("").astype(str)
        pieces.append(column + " " + values)

    return pd.Series(" ".join(row) for row in zip(*pieces))


def build_history_features(df, train_max_values=None):
    features = df[HISTORY_COLUMNS].fillna(0).astype(float).values
    features = np.log1p(features)

    if train_max_values is None:
        train_max_values = np.maximum(features.max(axis=0, keepdims=True), 1)

    return features / train_max_values, train_max_values


def explain_probability(score):
    if score < 0.20:
        return "very likely incorrect"
    if score < 0.40:
        return "probably incorrect"
    if score < 0.60:
        return "uncertain or mixed"
    if score < 0.80:
        return "probably correct"
    return "very likely correct"


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"

    train_df = pd.read_csv(data_dir / "train.tsv", sep="\t", names=COLUMNS)
    valid_df = pd.read_csv(data_dir / "valid.tsv", sep="\t", names=COLUMNS)
    test_df = pd.read_csv(data_dir / "test.tsv", sep="\t", names=COLUMNS)

    train_text = build_text_input(train_df)
    valid_text = build_text_input(valid_df)
    test_text = build_text_input(test_df)

    print("Building TF-IDF vocabulary from training data...")
    vectorizer = TFIDFVectorizer()
    vectorizer.build_vocab(train_text)

    X_train_text = normalize_rows(vectorizer.transform(train_text))
    X_valid_text = normalize_rows(vectorizer.transform(valid_text))
    X_test_text = normalize_rows(vectorizer.transform(test_text))

    X_train_history, train_max_values = build_history_features(train_df)
    X_valid_history, _ = build_history_features(valid_df, train_max_values)
    X_test_history, _ = build_history_features(test_df, train_max_values)

    X_train = np.hstack([X_train_text, X_train_history])
    X_valid = np.hstack([X_valid_text, X_valid_history])
    X_test = np.hstack([X_test_text, X_test_history])

    y_train = labels_to_binary(train_df["label"])
    y_valid = labels_to_binary(valid_df["label"])
    y_test = labels_to_binary(test_df["label"])

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Vocabulary size:  {X_train.shape[1]}")
    print(f"True-ish train labels: {np.mean(y_train) * 100:.2f}%")
    print("\nTraining binary truth MLP...")

    model = BinaryTruthMLP(
        input_size=X_train.shape[1],
        hidden_size=64,
        learning_rate=0.05,
        epochs=100,
        batch_size=128,
        patience=15,
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    test_scores = model.predict_proba(X_test)
    test_loss = model.loss(test_scores, y_test)
    test_acc_default = accuracy(test_scores, y_test, threshold=0.5)
    test_acc_tuned = accuracy(test_scores, y_test, threshold=model.best_threshold)

    print(f"\nFinal test loss: {test_loss:.4f}")
    print(f"Final test accuracy at 0.50 threshold: {test_acc_default * 100:.2f}%")
    print(
        f"Final test accuracy at validation-tuned threshold "
        f"({model.best_threshold:.2f}): {test_acc_tuned * 100:.2f}%"
    )

    print("\nExample predictions:")
    for index in range(5):
        statement = test_df["statement"].iloc[index]
        actual_label = test_df["label"].iloc[index]
        score = test_scores[index]
        print(f"\nStatement: {statement}")
        print(f"Actual label: {actual_label}")
        print(f"Probability true-ish: {score:.2f}")
        print(f"Meaning: {explain_probability(score)}")


if __name__ == "__main__":
    main()
