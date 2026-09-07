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
import csv
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

DATA_DIR = Path(__file__).resolve().parent / "data"


def read_liar(path):
    """Read a LIAR split. Use this, never a bare `pd.read_csv(..., sep='\\t')`.

    WHY IT EXISTS — THE DEFAULT PARSER SILENTLY LOSES ROWS:
    pandas defaults to `quotechar='"'`. LIAR is a tab-separated file whose
    statements quote people, so a claim containing one `"` opens a quoted
    field that stays open until the next `"` — several lines later. Everything
    in between is absorbed into that row's `statement`, carrying raw tab
    characters and the following rows' ids, labels and speaker metadata with
    it, under the FIRST row's label.

    Measured against the files on disk:

        split   lines in file   rows pandas returned   lost
        train      10,269              10,240           29
        valid       1,284               1,284            0
        test        1,283               1,267           16

    So the test set every number in this project was reported on had 1,267 of
    its 1,283 rows, 2 of which were several claims glued together; the longest
    "statement" it produced was 431 words, where the real maximum is 48. The
    training set was missing 29 rows and mislabelling a couple more. It is a
    small effect, but it is the kind that quietly moves the second decimal
    place of every figure, and there is no reason to accept it.

    QUOTE_NONE is correct here rather than a workaround: in a TSV the tab is
    the delimiter, so a quote character carries no structural meaning at all.
    """
    return pd.read_csv(path, sep="\t", names=COLUMNS, quoting=csv.QUOTE_NONE)


def load_split(name, data_dir=None):
    return read_liar(Path(data_dir or DATA_DIR) / f"{name}.tsv")

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
        weight_decay=0.0,
        early_stopping_patience=None,
        early_stopping_warmup=15,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # L2 penalty on the weight matrices. 0.0 reproduces the original
        # behaviour: no regularisation at all, on 26,626 input dimensions and
        # roughly ten thousand training rows.
        self.weight_decay = weight_decay
        # Stop when validation loss has not improved for this many epochs, and
        # restore the best weights seen. None runs every epoch regardless of
        # what validation is doing.
        self.early_stopping_patience = early_stopping_patience
        # Epochs before stopping may fire at all. This network starts almost
        # flat — inputs are L2-normalised sparse rows (~24 non-zeros of
        # 26,626), so He initialisation scaled for dense inputs gives hidden
        # activations around 0.004 and outputs in 0.4957-0.5056. Validation
        # loss therefore sits at ln(2) for the first several epochs while the
        # weights climb out, and patience counted that flat stretch as "no
        # improvement": hidden_size 32 and 128 both stopped around epoch 11
        # and scored EXACTLY the majority-class baseline, 0.5202, having
        # learned nothing. A warmup makes stopping mean "stopped improving"
        # rather than "has not started yet".
        self.early_stopping_warmup = early_stopping_warmup

        # The threshold determines where we draw the line between False and True.
        # It defaults to 0.5 (50%), but we "tune" it during training to find the best cutoff.
        self.best_threshold = 0.5

        # One generator for BOTH weight init and batch shuffling. The shuffle
        # used np.random.permutation — the global RNG — so `seed` controlled
        # only the initial weights and every run trained on a different batch
        # order. Two runs of the same configuration could differ by more than
        # the effects being measured, which makes an experiment unreadable and
        # contradicts the README's "reproduce these numbers with".
        rng = np.random.default_rng(seed)
        self._rng = rng
        
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

    def fit(self, X, y, X_valid=None, y_valid=None, quiet=False):
        n_samples = X.shape[0]
        # Stopping is driven by validation LOSS, not accuracy: accuracy
        # moves in discrete jumps on 1284 rows, so a plateau looks like
        # an improvement. Loss is continuous and registers growing
        # overconfidence before any label actually flips.
        best_valid_loss = None
        rounds_without_improvement = 0
        best_weights = None
        valid_loss = None

        for epoch in range(1, self.epochs + 1):
            indices = self._rng.permutation(n_samples)
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
                # Weight decay is applied to the weight matrices only, never
                # the biases: penalising a bias just shifts the decision
                # boundary without reducing model capacity.
                if self.weight_decay:
                    dW2 = dW2 + self.weight_decay * self.W2
                    dW1 = dW1 + self.weight_decay * self.W1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            # Early stopping needs the VALIDATION loss every epoch. It does
            # not need the training-set forward pass that reporting does, and
            # on 10,240 rows by 26,626 features that pass costs more than the
            # epoch itself — forcing the whole reporting block every epoch to
            # get this check roughly tripled training time.
            if (self.early_stopping_patience is not None
                    and X_valid is not None and y_valid is not None):
                epoch_valid_loss = self.loss(self.predict_proba(X_valid), y_valid)
                if best_valid_loss is None or epoch_valid_loss < best_valid_loss - 1e-5:
                    best_valid_loss = epoch_valid_loss
                    rounds_without_improvement = 0
                    best_weights = (self.W1.copy(), self.b1.copy(),
                                    self.W2.copy(), self.b2.copy())
                else:
                    rounds_without_improvement += 1
                # The warmup gates the STOP decision, not the tracking. An
                # earlier version skipped this whole block during warmup, which
                # also meant no weights were recorded — so a run whose best
                # model arrived inside the warmup window (lr 0.5 peaks around
                # epoch 15) would have thrown it away and kept a later, worse
                # one. Track always; refuse to stop early.
                if (epoch > self.early_stopping_warmup
                        and rounds_without_improvement >= self.early_stopping_patience):
                    if not quiet:
                        print(f"  early stop at epoch {epoch} "
                              f"(best valid loss {best_valid_loss:.4f})")
                    break

            # Reporting. `quiet` has to gate the COMPUTATION, not just the
            # print: this block runs a forward pass over the entire training
            # set, which on 10,240 rows by 26,626 features costs more than the
            # epoch that produced it. A quiet sweep was paying for output it
            # then threw away — five-fold cross-validation of this model spent
            # about a fifth of its time here.
            should_report = not quiet and (epoch == 1 or epoch % 5 == 0)
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

        # Restore the weights that generalised best, not the ones the last
        # epoch happened to leave behind.
        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2 = best_weights

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
            "kind": "mlp",
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


class LinearTruthModel:
    """L2-regularised logistic regression on sparse TF-IDF. The shipped model.

    WHY THIS REPLACED THE HIDDEN LAYER:
    `BinaryTruthMLP` above is kept — it is what the project shipped and what
    the Model Comparison page reports — but it is not the best model for this
    feature space, and a sweep says so rather than an opinion.

    Measured on the LIAR validation split (1,284 rows; selection never touched
    test), best configuration of each family:

        this model, word+char TF-IDF          0.6480
        LinearSVC + Platt scaling (sklearn)   0.6449
        sk-learn MLPClassifier, 256 hidden    0.6425
        shipped BinaryTruthMLP                0.6262
        majority class                        0.5202

    Two things are doing the work, and neither is depth. First, *strong* L2:
    the hidden layer trained with no penalty at all on 26,626 dimensions over
    10,240 rows, and every family in the sweep peaked at its most-regularised
    setting. Second, character n-grams: "tax", "taxes" and "taxpayer" share no
    word feature and most of a character one.

    A hidden layer buys nothing here because there is very little interaction
    structure to find — the signal in a claim's wording is close to additive.
    Adding one back on the same features costs about two points, which is the
    honest reason it is gone.

    NOT A FACT-CHECKER. See the module docstring and README: this scores how
    a claim is *worded* against a corpus of rated political statements. The
    verdict never reads it.
    """

    def __init__(
        self,
        input_size,
        l2=1e-4,
        epochs=400,
        learning_rate=2.0,
        momentum=0.9,
        halve_every=150,
        seed=42,
    ):
        self.input_size = input_size
        self.l2 = l2
        self.epochs = epochs
        self.lr = learning_rate
        self.momentum = momentum
        # Step-size decay, expressed in epochs rather than as a rate, so the
        # schedule reads the same whatever `epochs` is set to.
        self.halve_every = halve_every
        self.seed = seed
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.best_threshold = 0.5

    # `seed` is carried for interface parity with BinaryTruthMLP and for the
    # artifact; full-batch training from a zero start has nothing random in it,
    # which is the point — two runs of this configuration are bit-identical.

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def _scores(self, X):
        if hasattr(X, "dot") and not isinstance(X, np.ndarray):
            return X.dot(self.weights) + self.bias      # SparseMatrix
        return np.asarray(X) @ self.weights + self.bias  # dense, one live row

    def loss(self, predicted, actual):
        predicted = np.clip(np.asarray(predicted).flatten(), 1e-9, 1 - 1e-9)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

    def fit(self, X, y, X_valid=None, y_valid=None, quiet=False):
        """Full-batch gradient descent with momentum. Full-batch, not
        mini-batch, because the objective is convex and the whole gradient is
        one sparse product — there is nothing to gain from noise, and it makes
        the run deterministic."""
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        velocity = np.zeros_like(self.weights)
        bias_velocity = 0.0

        for epoch in range(1, self.epochs + 1):
            predicted = self.sigmoid(self._scores(X))
            residual = (predicted - y) / n
            # The L2 penalty is on the weights only. Penalising the bias would
            # drag the decision boundary towards 0.5 without reducing capacity.
            gradient = X.transpose_dot(residual) + self.l2 * self.weights
            bias_gradient = residual.sum()

            step = self.lr * (0.5 ** (epoch / self.halve_every))
            velocity = self.momentum * velocity - step * gradient
            self.weights += velocity
            bias_velocity = self.momentum * bias_velocity - step * bias_gradient
            self.bias += bias_velocity

            if not quiet and (epoch == 1 or epoch % 50 == 0):
                message = (f"Epoch {epoch:03d} | loss: {self.loss(predicted, y):.4f} | "
                           f"train accuracy: {accuracy(predicted, y) * 100:.2f}%")
                if X_valid is not None and y_valid is not None:
                    valid = self.predict_proba(X_valid)
                    message += (f" | valid loss: {self.loss(valid, y_valid):.4f} | "
                                f"valid accuracy: {accuracy(valid, y_valid) * 100:.2f}%")
                print(message)

        if X_valid is not None and y_valid is not None:
            self.best_threshold, _ = find_best_threshold(
                self.predict_proba(X_valid), y_valid)

    def predict_proba(self, X):
        return self.sigmoid(self._scores(X)).flatten()

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.best_threshold
        return (self.predict_proba(X) >= threshold).astype(int)

    def state_dict(self):
        return {
            "kind": "linear",
            "input_size": self.input_size,
            "l2": self.l2,
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "momentum": self.momentum,
            "halve_every": self.halve_every,
            "seed": self.seed,
            "best_threshold": self.best_threshold,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_state_dict(cls, state):
        model = cls(
            input_size=state["input_size"],
            l2=state["l2"],
            epochs=state["epochs"],
            learning_rate=state["learning_rate"],
            momentum=state.get("momentum", 0.9),
            halve_every=state.get("halve_every", 150),
            seed=state.get("seed", 42),
        )
        model.weights = state["weights"]
        model.bias = state["bias"]
        model.best_threshold = state["best_threshold"]
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

WHY IT SKIPS EMPTY FIELDS — THIS WAS A TRAIN/SERVE SKEW:
The previous version emitted every column unconditionally, name first:

    "statement The economy grew by 3 percent. subject  speaker  job  state  party  context "

`main()` trained the shipped model on the bare statement, but every live
request goes through `make_prediction_features()`, which fills the metadata
with blanks and passes the row through here. So the model was trained on one
string and served another: seven extra tokens plus seven boundary bigrams on
every single request, against a vocabulary that had never seen that shape.
It cost 0.47 points of accuracy — the model scores 62.35% on the text it was
trained on and 61.88% on the text production actually sends it.

Emitting a field only when it has a value makes a statement-only row come out
as exactly the statement, so the two paths cannot diverge again. The column
name is still prefixed to metadata fields, because "texas" as a state and
"texas" inside a claim are different features; the statement needs no tag
because it is the one field that is always present.
"""
def build_text_input(df):
    rows = []
    for _, row in df[TEXT_FEATURE_COLUMNS].iterrows():
        pieces = []
        for column in TEXT_FEATURE_COLUMNS:
            value = "" if pd.isna(row[column]) else str(row[column]).strip()
            if not value:
                continue
            pieces.append(value if column == "statement" else f"{column} {value}")
        rows.append(" ".join(pieces))

    return pd.Series(rows, dtype=object)

"""
PURPOSE: Extracts numerical history data and scales it down.
"""
# Which credit-history column each label increments. Note there is no column
# for "true" — LIAR ships five counts for six labels.
LABEL_TO_HISTORY_COLUMN = {
    "barely-true": "barely_true",
    "false": "false",
    "half-true": "half_true",
    "mostly-true": "mostly_true",
    "pants-fire": "pants_fire",
}


def build_history_features(df, train_max_values=None, deleak_labels=None):
    """The speaker's credit history, optionally with the current row removed.

    WHY `deleak_labels` EXISTS — THIS FEATURE LEAKS THE TARGET:
    The credit-history counts are meant to be the speaker's *past* record. They
    include the current statement's own label. Measured over the 2,054 speakers
    who appear exactly once in the whole dataset:

        own-label count == 1    99.2%
        total history  == 1     98.9%
        total history  == 0      0.8%

    A speaker with one statement carries exactly one count, sitting in that
    statement's own label column. The feature partly IS the target, and any
    accuracy trained on it is inflated — which is where this project's old
    "72.38% on LIAR" came from.

    There is a second-order version of the same leak. LIAR has no `true`
    column, so a solo speaker labelled "true" has all-zero history: 352 of the
    353 such rows. All-zero history was therefore itself a signal for "true".
    Subtracting the current row removes both — a solo "false" speaker also
    becomes all-zero, and the asymmetry goes with it.

    Passing ``deleak_labels`` (the split's label column) makes the feature mean
    "this speaker's OTHER statements", which is what it was always supposed to
    mean. Inference passes nothing: a live claim is not in anyone's history, so
    there is nothing to subtract.
    """
    features = df[HISTORY_COLUMNS].fillna(0).astype(float).values

    if deleak_labels is not None:
        column_index = {name: i for i, name in enumerate(HISTORY_COLUMNS)}
        for row, label in enumerate(deleak_labels):
            column = LABEL_TO_HISTORY_COLUMN.get(label)
            if column is not None:
                features[row, column_index[column]] -= 1.0
        # A handful of rows carry a zero count for their own label already;
        # clamp rather than let a negative through into log1p.
        features = np.maximum(features, 0.0)

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
    # Bundle everything needed to make a prediction into one object.
    #
    # Every vectorizer setting that changes the FEATURES has to be written
    # here. A setting that is saved by the trainer and defaulted by the loader
    # silently serves a different feature space than it trained on, which is
    # the same class of bug as the train/serve text skew in build_text_input().
    artifacts = {
        "model": model.state_dict(),
        "vectorizer": {
            "vocab": vectorizer.vocab,
            "idf_values": vectorizer.idf_values,
            "vocab_size": vectorizer.vocab_size,
            "ngram_range": vectorizer.ngram_range,
            "min_df": vectorizer.min_df,
            "char_ngram_range": vectorizer.char_ngram_range,
            "sublinear_tf": vectorizer.sublinear_tf,
            "smooth_idf": vectorizer.smooth_idf,
        },
        "train_max_values": train_max_values,
    }

    # Save it to disk using Python's "pickle" library
    with open(path, "wb") as file:
        pickle.dump(artifacts, file)


# Artifacts written before the linear model existed carry no "kind", and they
# are all MLPs.
MODEL_KINDS = {"mlp": BinaryTruthMLP, "linear": LinearTruthModel}


def load_artifacts(path):
    with open(path, "rb") as file:
        artifacts = pickle.load(file)

    # Reconstruct the TFIDF Vectorizer. `.get` with the pre-existing default
    # for each new setting, so an artifact saved by an older version of this
    # file still loads and still behaves exactly as it did.
    vectorizer_state = artifacts["vectorizer"]
    vectorizer = TFIDFVectorizer(
        ngram_range=vectorizer_state["ngram_range"],
        min_df=vectorizer_state["min_df"],
        char_ngram_range=vectorizer_state.get("char_ngram_range"),
        sublinear_tf=vectorizer_state.get("sublinear_tf", False),
        smooth_idf=vectorizer_state.get("smooth_idf", False),
    )
    vectorizer.vocab = vectorizer_state["vocab"]
    vectorizer.idf_values = vectorizer_state["idf_values"]
    vectorizer.vocab_size = vectorizer_state["vocab_size"]

    # Reconstruct whichever model was saved
    model_state = artifacts["model"]
    model = MODEL_KINDS[model_state.get("kind", "mlp")].from_state_dict(model_state)

    return model, vectorizer, artifacts["train_max_values"]

def build_feature_frame(statements, **metadata):
    """The one row shape every path builds from: the statement, whatever
    metadata the caller supplied, and the blank/zero defaults a live request
    sends for the rest."""
    rows = []
    for statement in statements:
        row = {"statement": statement}
        for column in TEXT_FEATURE_COLUMNS:
            if column != "statement":
                row[column] = metadata.get(column, "")
        for column in HISTORY_COLUMNS:
            row[column] = metadata.get(column, 0)
        rows.append(row)
    return pd.DataFrame(rows, columns=TEXT_FEATURE_COLUMNS + HISTORY_COLUMNS)


def make_training_features(vectorizer, statements, **metadata):
    """The same features as `make_prediction_features_batch`, sparse.

    WHY BOTH EXIST: a live request scores one row, where dense is simplest and
    cheapest. Training scores 10,240 rows against 74,429 features, where dense
    is 6.1 GB of mostly zeros. The two must agree exactly or the model is
    trained on something other than what it is served —
    `tests/test_model_evaluation_path.py` pins that they do, row by row.

    The history columns are appended as empty columns rather than dropped, so
    a weight vector trained here indexes the same features a dense serving row
    presents. They carry no values because a live claim has no speaker history
    (and, per `build_history_features`, the training-set version of that
    feature leaks the label anyway).
    """
    frame = build_feature_frame(statements, **metadata)
    matrix = vectorizer.transform_sparse(build_text_input(frame), l2_normalize=True)
    matrix.n_columns += len(HISTORY_COLUMNS)
    return matrix


def predict_proba_texts(model, vectorizer, train_max_values, statements, chunk=128):
    """Score many statements through the live request's own feature path.

    Evaluation has to use the serving path or its number describes a model
    nobody runs — that is how this project once reported 56.9% for a model
    that scores 61.9%. But the dense serving form of 1,267 test rows against
    74,429 features is 754 MB, so it goes through in chunks.
    """
    statements = list(statements)
    scores = []
    for start in range(0, len(statements), chunk):
        features = make_prediction_features_batch(
            vectorizer, train_max_values, statements[start:start + chunk])
        scores.append(model.predict_proba(features))
    return np.concatenate(scores) if scores else np.zeros(0)


"""
PURPOSE: Core function used by `main.py` to turn raw text into model-ready numbers.
"""
def make_prediction_features_batch(
    vectorizer,
    train_max_values,
    statements,
    **metadata,
):
    """Features for many statements, built exactly as a live request builds them.

    WHY THIS EXISTS:
    Evaluation used to construct its own features. `evaluate_models.py` loaded
    the shipped model and then scored it on `build_text_input(test_df)` — the
    statement PLUS subject, speaker, job, state, party and context — with real
    non-zero history counts. The shipped model is trained statement-only with
    history zeroed (see `main()` below), so it was being measured on a
    distribution it had never seen, and reported **56.9%** for a model that
    scores **61.9%** on the inputs it actually receives.

    Everything that scores this model now goes through here, so evaluation and
    serving cannot drift apart again. Any metadata a caller does not supply
    defaults to the blank/zero value the API sends.
    """
    frame = build_feature_frame(statements, **metadata)

    # 1. Process Text
    text_features = normalize_rows(vectorizer.transform(build_text_input(frame)))

    # 2. Process History
    history_features, _ = build_history_features(frame, train_max_values)

    # 3. Glue them together side-by-side
    return np.hstack([text_features, history_features])


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
    """One statement's features. Delegates so there is a single code path."""
    return make_prediction_features_batch(
        vectorizer,
        train_max_values,
        [statement],
        subject=subject,
        speaker=speaker,
        job=job,
        state=state,
        party=party,
        context=context,
        barely_true=barely_true,
        false=false,
        half_true=half_true,
        mostly_true=mostly_true,
        pants_fire=pants_fire,
    )


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

# The configuration `train_experiments.py` selected by 5-fold cross-validation
# over train+valid. Nothing here was chosen by looking at the test set.
#
# WHY NO CHARACTER N-GRAMS, despite them looking good at first:
# On the single 1,284-row validation split, adding char 3-5grams was worth
# +0.6 points and the best AUC in the sweep. Cross-validated over 11,553 rows
# it is worth NOTHING -- 0.6254 against 0.6263 for word features alone -- and
# it triples the vocabulary. The apparent gain was the split. That is the
# entire reason selection moved to cross-validation; see the table in
# docs/ML_MODEL_INVESTIGATION.md.
SHIPPED_VECTORIZER = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "smooth_idf": True,
}
SHIPPED_MODEL = {"l2": 1e-4, "epochs": 400, "learning_rate": 2.0}


def main():
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df = load_split("test")

    # Train on the same statement-only input available to the live API.
    # Including speaker history here would inflate offline scores because
    # production requests do not provide those fields.
    train_text = train_df["statement"].fillna("").astype(str)
    valid_text = valid_df["statement"].fillna("").astype(str)
    test_text = test_df["statement"].fillna("").astype(str)

    print("Building TF-IDF vocabulary from training data...")
    vectorizer = TFIDFVectorizer(**SHIPPED_VECTORIZER)
    vectorizer.build_vocab(build_text_input(build_feature_frame(train_text)))

    # Features built by the same function the live request uses, so the model
    # cannot be trained on a different string than it is served.
    X_train = make_training_features(vectorizer, train_text)
    X_valid = make_training_features(vectorizer, valid_text)
    X_test = make_training_features(vectorizer, test_text)

    # History counts are zero everywhere: production never sends them, and the
    # training-set version of the feature leaks the label (see
    # build_history_features). `train_max_values` exists only so the serving
    # path has something to divide by.
    train_max_values = np.ones((1, len(HISTORY_COLUMNS)))

    y_train = labels_to_binary(train_df["label"])
    y_valid = labels_to_binary(valid_df["label"])
    y_test = labels_to_binary(test_df["label"])

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Feature count:    {X_train.shape[1]}")
    print(f"True-ish train labels: {np.mean(y_train) * 100:.2f}%")
    print("\nTraining regularised linear truth model...")

    model = LinearTruthModel(input_size=X_train.shape[1], **SHIPPED_MODEL)
    model.fit(X_train, y_train, X_valid, y_valid)

    save_artifacts(MODEL_FILE, model, vectorizer, train_max_values)
    print(f"\nSaved model artifacts to: {MODEL_FILE}")

    test_scores = model.predict_proba(X_test)
    test_loss = model.loss(test_scores, y_test)
    test_acc_default = accuracy(test_scores, y_test, threshold=0.5)
    test_acc_tuned = accuracy(test_scores, y_test, threshold=model.best_threshold)
    baseline = max(float(y_test.mean()), 1 - float(y_test.mean()))

    print(f"\nFinal test loss: {test_loss:.4f}")
    print(f"Final test accuracy at 0.50 threshold: {test_acc_default * 100:.2f}%")
    print(
        f"Final test accuracy at validation-tuned threshold "
        f"({model.best_threshold:.2f}): {test_acc_tuned * 100:.2f}%"
    )
    print(f"Majority-class baseline on the same rows: {baseline * 100:.2f}%")

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
