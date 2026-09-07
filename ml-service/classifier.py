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
    """
    WHY THE DEFAULTS CHANGED — THE OLD ONES NEVER TRAINED:

    This ran 100 full-batch steps at lr=0.01 (and 0.1 from evaluate_models.py)
    from a zero start, on L2-normalised TF-IDF rows. Measured on the LIAR test
    set, the model it produced emitted scores in the range [0.5543, 0.5597]:
    every single one above 0.5, so it predicted "true-ish" for all 1,267 rows
    and scored 56.35% — exactly the majority-class rate, to four decimals.

    That is not a weak baseline, it is a *stopped* one, and it made the Model
    Comparison page misleading in the direction that flatters the production
    model. The tell was in the same run: its ROC-AUC was 0.661, so the ranking
    it had learned was fine. Only the decision boundary had never moved.

    Full-batch gradient descent on this objective needs a step size in the
    ones, not the hundredths, because the L2-normalised rows make the gradient
    tiny. With momentum and enough epochs it converges, and the number it then
    reports is a real lower bound on what the neural network has to beat.
    """

    def __init__(self, learning_rate=2.0, epochs=400, l2=1e-4, momentum=0.9):
        self.lr = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.momentum = momentum
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        # converts any number to 0-1 probability
        return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

    def fit(self, X, y, quiet=False):
        # X is shape (num_statements, vocab_size)
        # y is shape (num_statements,) — 0 or 1
        num_samples, num_features = X.shape

        # start weights at zero
        self.weights = np.zeros(num_features)
        self.bias = 0
        velocity = np.zeros(num_features)
        bias_velocity = 0.0

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
            dw = np.dot(X.T, error) / num_samples + self.l2 * self.weights
            db = np.mean(error)

            # Step decay, so the run can start fast and still settle.
            step = self.lr * (0.5 ** (epoch / 150))
            velocity = self.momentum * velocity - step * dw
            self.weights += velocity
            bias_velocity = self.momentum * bias_velocity - step * db
            self.bias += bias_velocity

            if not quiet and epoch % 50 == 0:
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

    # load data -- through read_liar(), because pandas' default quote handling
    # swallows 45 rows across the three splits. See binary_truth_mlp.read_liar.
    from binary_truth_mlp import load_split

    train_df = load_split("train")
    test_df = load_split("test")

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
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # evaluate
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")