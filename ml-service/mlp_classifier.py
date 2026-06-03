"""
FILE PURPOSE:
This file defines a basic Neural Network (Multi-Layer Perceptron or MLP) for 6-class text classification.
It tries to predict the exact rating (e.g., "pants-fire", "mostly-true") of a statement.

FLOW:
1. Defines the network architecture (1 hidden layer).
2. Uses Forward Propagation to make guesses.
3. Uses Backpropagation to learn from its mistakes.
4. `main()` loads data, trains the model, and prints accuracy.

USED BY:
- This is primarily an experimental/research script. 
- The main application actually uses `binary_truth_mlp.py` because predicting between 6 highly-subjective classes is much harder than a simple True/False binary prediction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tfidf import TFIDFVectorizer

# The 6 possible labels in the LIAR dataset
LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire",
    "context",
]

class MLPClassifier:
    """
    PURPOSE: Initialize the Neural Network with random weights.
    
    INPUT:
    input_size: How many numbers represent one sentence (Vocab size).
    hidden_size: How many "neurons" are in the middle brain.
    output_size: 6 (because there are 6 possible labels).
    """
    def __init__(
        self,
        input_size,
        hidden_size=64,
        output_size=6,
        learning_rate=0.03,
        epochs=30,
        batch_size=128,
        seed=42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        rng = np.random.default_rng(seed)
        
        # W1 and b1 connect the Input -> Hidden Layer
        # We use a mathematical trick called "He initialization" (sqrt(2 / input_size)) 
        # so the random numbers aren't too big or too small.
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        # W2 and b2 connect the Hidden Layer -> Output Layer
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    """
    PURPOSE: Activation function that turns negative numbers to 0.
    WHY: Neural networks need non-linear math to learn complex patterns.
    """
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    """
    PURPOSE: Converts raw output numbers into 6 probabilities that add up to 1.0 (100%).
    """
    def softmax(self, z):
        # Shift to prevent numbers from getting too huge and crashing Python (overflow)
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    """
    PURPOSE: Converts a single label (e.g., 2) into an array (e.g., [0, 0, 1, 0, 0, 0]).
    WHY: The neural network outputs 6 numbers, so we need the "correct answer" to also be 6 numbers to compare them.
    """
    def one_hot(self, y):
        encoded = np.zeros((len(y), self.output_size))
        encoded[np.arange(len(y)), y] = 1
        return encoded

    """
    PURPOSE: Makes a prediction. (Data moves "Forward" through the network)
    """
    def forward(self, X):
        # 1. Multiply inputs by weights
        z1 = np.dot(X, self.W1) + self.b1
        # 2. Apply activation
        a1 = self.relu(z1)
        # 3. Multiply hidden layer by weights
        z2 = np.dot(a1, self.W2) + self.b2
        # 4. Turn into probabilities
        probabilities = self.softmax(z2)
        return z1, a1, probabilities

    """
    PURPOSE: Calculates how "wrong" the model's guesses were (Cross-Entropy Loss).
    """
    def loss(self, probabilities, y):
        samples = len(y)
        # Grab the probability the model assigned to the CORRECT class
        correct_probs = probabilities[np.arange(samples), y]
        return -np.mean(np.log(correct_probs + 1e-9)) # +1e-9 prevents log(0) crashing

    """
    PURPOSE: The Training Loop. Where the model actually learns.
    """
    def fit(self, X, y, X_valid=None, y_valid=None):
        y_one_hot = self.one_hot(y)
        n_samples = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            # Step 1: Shuffle the data every time so it doesn't memorize the order
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            y_one_hot_shuffled = y_one_hot[indices]

            # Step 2: Train in small chunks (batches) instead of all at once. It's faster.
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_batch_one_hot = y_one_hot_shuffled[start:end]
                batch_size = X_batch.shape[0]

                # --- FORWARD PASS (Make guesses) ---
                z1, a1, probabilities = self.forward(X_batch)

                # --- BACKWARD PASS (Figure out mistakes using Calculus) ---
                # How wrong was the output?
                dz2 = (probabilities - y_batch_one_hot) / batch_size
                # How much should W2 change?
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                # How much of the blame goes to the hidden layer?
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self.relu_derivative(z1)
                # How much should W1 change?
                dW1 = np.dot(X_batch.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # --- UPDATE WEIGHTS (Actually "learning") ---
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            # Print progress every 5 epochs
            if epoch == 1 or epoch % 5 == 0:
                sample_size = min(1000, X.shape[0])
                sample_X = X[:sample_size]
                sample_y = y[:sample_size]
                train_probs = self.predict_proba(sample_X)
                train_loss = self.loss(train_probs, sample_y)
                train_acc = accuracy(self.predict(sample_X), sample_y)

                message = (
                    f"Epoch {epoch:03d} | "
                    f"loss: {train_loss:.4f} | "
                    f"train accuracy: {train_acc * 100:.2f}%"
                )

                if X_valid is not None and y_valid is not None:
                    valid_acc = accuracy(self.predict(X_valid), y_valid)
                    message += f" | valid accuracy: {valid_acc * 100:.2f}%"

                print(message)

    def predict_proba(self, X):
        _, _, probabilities = self.forward(X)
        return probabilities

    def predict(self, X):
        # Pick the label with the highest probability
        return np.argmax(self.predict_proba(X), axis=1)


# --- Helper Functions ---

def labels_to_numbers(labels):
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    return labels.map(label_to_id).values

def accuracy(predictions, actual):
    return np.mean(predictions == actual)

def normalize_rows(X):
    # Scales each row so the vector length is 1. Helps the model learn stably.
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def print_class_accuracy(predictions, actual):
    print("\nAccuracy by label:")
    for index, label in enumerate(LABELS):
        mask = actual == index
        if np.sum(mask) == 0:
            continue
        class_acc = accuracy(predictions[mask], actual[mask])
        print(f"  {label:<12} {class_acc * 100:6.2f}% ({np.sum(mask)} samples)")


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"

    train_df = pd.read_csv(data_dir / "train.tsv", sep="\t", names=COLUMNS)
    valid_df = pd.read_csv(data_dir / "valid.tsv", sep="\t", names=COLUMNS)
    test_df = pd.read_csv(data_dir / "test.tsv", sep="\t", names=COLUMNS)

    print("Building TF-IDF vocabulary from training data...")
    vectorizer = TFIDFVectorizer()
    vectorizer.build_vocab(train_df["statement"])

    X_train = normalize_rows(vectorizer.transform(train_df["statement"]))
    X_valid = normalize_rows(vectorizer.transform(valid_df["statement"]))
    X_test = normalize_rows(vectorizer.transform(test_df["statement"]))

    y_train = labels_to_numbers(train_df["label"])
    y_valid = labels_to_numbers(valid_df["label"])
    y_test = labels_to_numbers(test_df["label"])

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Vocabulary size:  {X_train.shape[1]}")
    print(f"Classes:          {len(LABELS)}")
    print("\nTraining MLP...")

    model = MLPClassifier(
        input_size=X_train.shape[1],
        hidden_size=64,
        output_size=len(LABELS),
        learning_rate=0.03,
        epochs=30,
        batch_size=128,
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    predictions = model.predict(X_test)
    test_accuracy = accuracy(predictions, y_test)

    print(f"\nFinal test accuracy: {test_accuracy * 100:.2f}%")
    print_class_accuracy(predictions, y_test)

if __name__ == "__main__":
    main()
