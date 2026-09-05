"""
FILE PURPOSE:
The simplest baseline model in the project: a from-scratch logistic
regression classifier (one linear layer + sigmoid, no hidden layer),
trained on TF-IDF bigrams only. It exists purely as a lower bound to
compare the MLP models against — "how well does a straight line through
the data do?"

USED BY:
- evaluate_models.py (offline comparison script that powers the numbers
  shown on the frontend's Model Comparison page). Not used by the live
  API — main.py loads binary_truth_mlp.py instead.

NOT to be confused with the production model: this file is experimental/
research-only, kept for the "Logistic Regression" column in Model Comparison.
"""

import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        # converts any number to 0-1 probability
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # X is shape (num_statements, vocab_size)
        # y is shape (num_statements,) — 0 or 1
        num_samples, num_features = X.shape

        # start weights at zero
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # forward pass — make predictions
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # how wrong are we? (loss)
            loss = -np.mean(
                y * np.log(predictions + 1e-9) +
                (1 - y) * np.log(1 - predictions + 1e-9)
            )

            # gradient descent — nudge weights in the right direction
            error = predictions - y
            dw = np.dot(X.T, error) / num_samples
            db = np.mean(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 10 == 0:
                print(f"Epoch {epoch} — loss: {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.path.append('..')
    from model.tfidf import TFIDFVectorizer

    columns = [
        "id", "label", "statement", "subject", "speaker",
        "job", "state", "party",
        "barely_true", "false", "half_true", "mostly_true", "pants_fire",
        "context"
    ]

    # load data
    train_df = pd.read_csv('../data/train.tsv', sep='\t', names=columns)
    test_df  = pd.read_csv('../data/test.tsv',  sep='\t', names=columns)

    # simplify labels to binary
    fake = {'pants-fire', 'false', 'barely-true'}
    train_df['binary'] = train_df['label'].apply(lambda x: 0 if x in fake else 1)
    test_df['binary']  = test_df['label'].apply(lambda x: 0 if x in fake else 1)

    # build TF-IDF vectors
    print("Building TF-IDF vectors...")
    vectorizer = TFIDFVectorizer()
    vectorizer.build_vocab(train_df['statement'])

    X_train = vectorizer.transform(train_df['statement'])
    X_test  = vectorizer.transform(test_df['statement'])
    from numpy.linalg import norm
    X_train = X_train / (norm(X_train, axis=1, keepdims=True) + 1e-9)
    X_test  = X_test  / (norm(X_test,  axis=1, keepdims=True) + 1e-9)
    y_train = train_df['binary'].values
    y_test  = test_df['binary'].values

    print(f"Training on {X_train.shape[0]} statements...")

    # train the model
    model = LogisticRegression(learning_rate=0.1, epochs=100)
    model.fit(X_train, y_train)

    # evaluate
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")