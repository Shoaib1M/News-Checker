"""
FILE PURPOSE:
This is the core Neural Network used in production by the API.
Instead of predicting 6 distinct labels (which is very hard), it groups them into 2 categories:
"Fake-ish" (pants-fire, false, barely-true) vs "True-ish" (half-true, mostly-true, true).

FLOW:
1. `BinaryTruthMLP`: A Neural Network that predicts a single probability between 0 and 1.
2. `make_prediction_features()`: Combines the text of the statement with historical data (like a politician's past truth record) into one giant array of numbers.
3. `fit()`: The training loop (Forward pass + Backpropagation).
4. `save_artifacts() / load_artifacts()`: Saves the trained "brain" to the hard drive so the web server can load it instantly.

USED BY:
- `main.py` uses `load_artifacts()`, `make_prediction_features()`, and `predict_proba()` to answer live web requests.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from tfidf import TFIDFVectorizer

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire",
    "context",
]

# We collapse 6 categories into 2 simple buckets for binary classification
FAKEISH_LABELS = {"pants-fire", "false", "barely-true"}
TRUEISH_LABELS = {"half-true", "mostly-true", "true"}

# The text fields we want the model to read
TEXT_FEATURE_COLUMNS = ["statement", "subject", "speaker", "job", "state", "party", "context"]

# The historical truth record of the speaker (how many times they've lied in the past)
HISTORY_COLUMNS = ["barely_true", "false", "half_true", "mostly_true", "pants_fire"]

# Where we save the trained model on disk
MODEL_FILE = Path(__file__).resolve().parent / "binary_truth_mlp.pkl"


class BinaryTruthMLP:
    """
    PURPOSE: Initialize the Neural Network for Binary Classification.
    Notice the output_size is missing, because a binary classifier just needs 1 output node 
    (a percentage from 0 to 1).
    """
    def __init__(
        self,
        input_size,
        hidden_size=64,
        learning_rate=0.05,
        epochs=40,
        batch_size=128,
        seed=42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # The threshold determines where we draw the line between False and True.
        # It defaults to 0.5 (50%), but we "tune" it during training to find the best cutoff.
        self.best_threshold = 0.5

        rng = np.random.default_rng(seed)
        
        # Layer 1 (Input -> Hidden)
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        # Layer 2 (Hidden -> Output: just 1 node)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    # Activation function for hidden layer
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    """
    PURPOSE: Squashes any number into a range between exactly 0.0 and 1.0.
    WHY: Perfect for calculating probabilities!
    """
    def sigmoid(self, z):
        z = np.clip(z, -500, 500) # Prevent math crash if z is massively negative/positive
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probability = self.sigmoid(z2) # Use sigmoid instead of softmax for binary choice
        return z1, a1, probability

    """
    PURPOSE: Calculates Binary Cross-Entropy Loss. 
    It heavily penalizes the model if it is extremely confident but WRONG.
    """
    def loss(self, predicted, actual):
        predicted = np.clip(predicted.flatten(), 1e-9, 1 - 1e-9)
        return -np.mean(
            actual * np.log(predicted) +
            (1 - actual) * np.log(1 - predicted)
        )

    def fit(self, X, y, X_valid=None, y_valid=None):
        n_samples = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices].reshape(-1, 1)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                current_batch_size = X_batch.shape[0]

                # --- FORWARD PASS ---
                z1, a1, predicted = self.forward(X_batch)

                # --- BACKWARD PASS (Calculus to find mistakes) ---
                # For sigmoid + binary cross-entropy, this gradient simplifies nicely.
                dz2 = (predicted - y_batch) / current_batch_size
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self.relu_derivative(z1)
                dW1 = np.dot(X_batch.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # --- UPDATE WEIGHTS ---
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            # Reporting
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

        # After training finishes, find the absolute best cutoff line based on validation data
        if X_valid is not None and y_valid is not None:
            valid_pred = self.predict_proba(X_valid)
            self.best_threshold, _ = find_best_threshold(valid_pred, y_valid)

    def predict_proba(self, X):
        _, _, probability = self.forward(X)
        return probability.flatten()

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        return (self.predict_proba(X) >= threshold).astype(int)

    """
    PURPOSE: Packages up the trained weights and configurations into a dictionary.
    WHY: So we can save it to a file.
    """
    def state_dict(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "best_threshold": self.best_threshold,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

    """
    PURPOSE: Recreates the model from a loaded dictionary of weights.
    """
    @classmethod
    def from_state_dict(cls, state):
        model = cls(
            input_size=state["input_size"],
            hidden_size=state["hidden_size"],
            learning_rate=state["learning_rate"],
            epochs=state["epochs"],
            batch_size=state["batch_size"],
        )
        model.best_threshold = state["best_threshold"]
        model.W1 = state["W1"]
        model.b1 = state["b1"]
        model.W2 = state["W2"]
        model.b2 = state["b2"]
        return model


# ---------------------------------------------------------------------------
# DATA PREPARATION HELPERS
# ---------------------------------------------------------------------------

"""
PURPOSE: Converts string labels ("pants-fire", "true") into numbers (0.0 or 1.0).
"""
def labels_to_binary(labels):
    return labels.apply(lambda label: 1 if label in TRUEISH_LABELS else 0).values.astype(float)


def accuracy(predicted_scores, actual, threshold=0.5):
    predicted = predicted_scores >= threshold
    return np.mean(predicted == actual)

"""
PURPOSE: Finds the optimal decision cutoff.
WHY: Sometimes the model is hesitant. E.g., maybe it never gives a score higher than 40%.
By tuning the threshold (e.g., deciding anything > 35% is True), we can maximize real-world accuracy.
"""
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

"""
PURPOSE: Mashes the statement, speaker name, job, and state into one long string.
"""
def build_text_input(df):
    pieces = []
    for column in TEXT_FEATURE_COLUMNS:
        values = df[column].fillna("").astype(str)
        pieces.append(column + " " + values)

    return pd.Series(" ".join(row) for row in zip(*pieces))

"""
PURPOSE: Extracts numerical history data and scales it down.
"""
def build_history_features(df, train_max_values=None):
    features = df[HISTORY_COLUMNS].fillna(0).astype(float).values
    # log1p prevents people with 10,000 past statements from overpowering the model
    features = np.log1p(features)

    # Scale everything so max is 1.0
    if train_max_values is None:
        train_max_values = np.maximum(features.max(axis=0, keepdims=True), 1)

    return features / train_max_values, train_max_values

# ---------------------------------------------------------------------------
# SAVING & LOADING
# ---------------------------------------------------------------------------

def save_artifacts(path, model, vectorizer, train_max_values):
    # Bundle everything needed to make a prediction into one object
    artifacts = {
        "model": model.state_dict(),
        "vectorizer": {
            "vocab": vectorizer.vocab,
            "idf_values": vectorizer.idf_values,
            "vocab_size": vectorizer.vocab_size,
            "ngram_range": vectorizer.ngram_range,
            "min_df": vectorizer.min_df,
        },
        "train_max_values": train_max_values,
    }

    # Save it to disk using Python's "pickle" library
    with open(path, "wb") as file:
        pickle.dump(artifacts, file)


def load_artifacts(path):
    with open(path, "rb") as file:
        artifacts = pickle.load(file)

    # Reconstruct the TFIDF Vectorizer
    vectorizer_state = artifacts["vectorizer"]
    vectorizer = TFIDFVectorizer(
        ngram_range=vectorizer_state["ngram_range"],
        min_df=vectorizer_state["min_df"],
    )
    vectorizer.vocab = vectorizer_state["vocab"]
    vectorizer.idf_values = vectorizer_state["idf_values"]
    vectorizer.vocab_size = vectorizer_state["vocab_size"]

    # Reconstruct the Neural Network
    model = BinaryTruthMLP.from_state_dict(artifacts["model"])
    
    return model, vectorizer, artifacts["train_max_values"]

"""
PURPOSE: Core function used by `main.py` to turn raw text into model-ready numbers.
"""
def make_prediction_features(
    vectorizer,
    train_max_values,
    statement,
    subject="",
    speaker="",
    job="",
    state="",
    party="",
    context="",
    barely_true=0,
    false=0,
    half_true=0,
    mostly_true=0,
    pants_fire=0,
):
    # Pack it into a 1-row DataFrame so our builder functions work normally
    row = pd.DataFrame([{
        "statement": statement,
        "subject": subject,
        "speaker": speaker,
        "job": job,
        "state": state,
        "party": party,
        "context": context,
        "barely_true": barely_true,
        "false": false,
        "half_true": half_true,
        "mostly_true": mostly_true,
        "pants_fire": pants_fire,
    }])

    # 1. Process Text
    text = build_text_input(row)
    text_features = normalize_rows(vectorizer.transform(text))
    
    # 2. Process History
    history_features, _ = build_history_features(row, train_max_values)
    
    # 3. Glue them together side-by-side
    return np.hstack([text_features, history_features])


def predict_statement(model, vectorizer, train_max_values, statement, **metadata):
    X = make_prediction_features(
        vectorizer=vectorizer,
        train_max_values=train_max_values,
        statement=statement,
        **metadata,
    )
    score = model.predict_proba(X)[0]
    predicted_class = "true-ish" if score >= model.best_threshold else "fake-ish"
    return score, predicted_class, explain_probability(score)


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


# ---------------------------------------------------------------------------
# LOCAL TRAINING & TESTING SCRIPTS
# ---------------------------------------------------------------------------

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
        epochs=70,
        batch_size=128,
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    save_artifacts(MODEL_FILE, model, vectorizer, train_max_values)
    print(f"\nSaved model artifacts to: {MODEL_FILE}")

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


def interactive_predict():
    if not MODEL_FILE.exists():
        print("No saved model found yet.")
        print("Train the model first by running: python binary_truth_mlp.py")
        return

    model, vectorizer, train_max_values = load_artifacts(MODEL_FILE)

    print("Loaded saved binary truth MLP.")
    print("Type a statement to score it. Press Enter on an empty line to quit.")
    print("Note: without speaker/topic metadata, this is a claim-only estimate.\n")

    while True:
        statement = input("Statement: ").strip()
        if not statement:
            break

        score, predicted_class, meaning = predict_statement(
            model,
            vectorizer,
            train_max_values,
            statement,
        )

        print(f"Probability true-ish: {score:.2f}")
        print(f"Decision threshold: {model.best_threshold:.2f}")
        print(f"Prediction: {predicted_class}")
        print(f"Meaning: {meaning}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        interactive_predict()
    else:
        main()
