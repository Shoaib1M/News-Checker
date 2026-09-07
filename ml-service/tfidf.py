"""
FILE PURPOSE:
This file defines the TFIDFVectorizer class.
TF-IDF stands for Term Frequency-Inverse Document Frequency.
It converts text into an array of numbers representing how "important" each word is to that specific sentence.

FLOW:
1. `clean()` & `get_ngrams()`: Breaks text into words and word-pairs (n-grams).
2. `build_vocab()`: Reads all training documents and calculates the IDF (Inverse Document Frequency) for every word.
3. `transform()`: Takes a new sentence and calculates its TF-IDF score vector.

USED BY:
- `binary_truth_mlp.py` (To train the neural network)
- `main.py` (To transform live user input before passing it to the model)
"""

import numpy as np
import math
import re

# Character n-grams live in the same vocabulary as word n-grams, so they need a
# prefix that no word n-gram can produce. `clean()` strips everything but
# [a-z0-9 ], so a leading NUL is unreachable for a word token.
CHAR_PREFIX = "\x00"


class TFIDFVectorizer:
    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=2,
        char_ngram_range=None,
        sublinear_tf=False,
        smooth_idf=False,
    ):
        self.vocab = {}          # Maps a token to its column index in the final vector
        self.idf_values = {}     # Stores the calculated IDF score for each token
        self.vocab_size = 0

        # ngram_range=(1, 2) means we look at single words (unigrams) AND pairs of words (bigrams).
        # Example for "fake news": Unigrams: ["fake", "news"]. Bigrams: ["fake news"].
        # Pass None to switch word n-grams off entirely (character-only model).
        self.ngram_range = ngram_range

        # min_df (Minimum Document Frequency): Ignore words that appear in fewer than 2 documents.
        # This filters out extremely rare words or typos to keep the vocabulary size manageable.
        self.min_df = min_df

        # Character n-grams, taken inside word boundaries. (3, 5) on "budget"
        # gives " bu", "bud", "udg", ... " budg", "udget", "dget ". They survive
        # the spelling variation, hyphenation and morphology that whole-word
        # features miss ("tax", "taxes", "taxpayer" share nothing as words), and
        # on this corpus they carry as much signal as the word features do -- see
        # the sweep in docs/ML_MODEL_INVESTIGATION.md. None keeps the original
        # word-only behaviour.
        self.char_ngram_range = char_ngram_range

        # tf = 1 + log(count) instead of count / total_tokens. The raw ratio
        # makes a term's weight depend on how long the sentence is, which is not
        # a property of the term; the log form is the standard fix and it is
        # what every strong baseline in the sweep used.
        self.sublinear_tf = sublinear_tf

        # idf = log((1 + N) / (1 + df)) + 1 instead of log(N / df). The +1 inside
        # avoids a zero-division on an unseen token and the +1 outside stops a
        # term that appears in every document from being annihilated (log(N/N)=0
        # deletes the feature rather than merely down-weighting it).
        self.smooth_idf = smooth_idf

    """
    PURPOSE: Standardizes the text.
    """
    def clean(self, text):
        text = str(text).lower()
        # Keep letters, numbers, and spaces. Replace everything else with a space.
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Collapse multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    """
    PURPOSE: Character n-grams of each word, padded so that prefixes and
    suffixes are distinguishable from the middle of a word.
    """
    def get_char_ngrams(self, words):
        if not self.char_ngram_range:
            return []
        min_n, max_n = self.char_ngram_range
        tokens = []
        for word in words:
            padded = f" {word} "
            for n in range(min_n, max_n + 1):
                if len(padded) < n:
                    continue
                for index in range(len(padded) - n + 1):
                    tokens.append(CHAR_PREFIX + padded[index:index + n])
        return tokens

    """
    PURPOSE: Every token this vectorizer knows how to produce for one document.
    """
    def tokenize(self, text):
        words = self.clean(text)
        tokens = self.get_ngrams(words) if self.ngram_range else []
        tokens.extend(self.get_char_ngrams(words))
        return tokens

    """
    PURPOSE: Generates unigrams and bigrams from a list of words.
    
    WHY THIS EXISTS:
    "Not good" means the opposite of "good". If we only look at single words, the model misses context.
    Bigrams capture pairs of words to preserve some order.
    """
    def get_ngrams(self, words):
        tokens = []
        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            if len(words) < n:
                continue

            for index in range(len(words) - n + 1):
                tokens.append(" ".join(words[index:index + n]))

        return tokens

    """
    PURPOSE: Calculates the Inverse Document Frequency (IDF) for all tokens across the entire dataset.
    
    WHY THIS EXISTS:
    Words like "the" or "is" appear in every document, so they aren't useful for classification.
    IDF mathematically penalizes words that appear everywhere, and rewards rare words that are highly specific.
    """
    def build_vocab(self, documents):
        total_docs = len(documents)

        # Step 1: Count how many documents contain each token
        doc_frequency = {}
        for doc in documents:
            tokens = set(self.tokenize(doc)) # Use set() so we only count a word once per document
            for token in tokens:
                doc_frequency[token] = doc_frequency.get(token, 0) + 1

        # Step 2: Calculate the IDF score for tokens that meet the minimum
        # frequency.
        #
        # SORTED, and that matters. Step 1 iterates a set of strings, and
        # Python randomises string hashing per process, so the insertion order
        # of `doc_frequency` differs between runs. Assigning column indices in
        # that order gave every token a different column each time the
        # vocabulary was built — the feature matrix was column-permuted per
        # process, seeded weights lined up against different tokens, and two
        # runs of an identical configuration produced different models.
        #
        # It was invisible in-process (one interpreter, one hash seed) and only
        # showed up as the same variant scoring 0.6402 and then 0.6379 across
        # two invocations of the experiment sweep. Sorting gives a canonical
        # column order that does not depend on the run.
        for token, count in sorted(doc_frequency.items()):
            if count >= self.min_df:
                # Assign this token a permanent index/column in our vectors
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1

                # Formula for IDF: log( Total Documents / Documents containing word )
                if self.smooth_idf:
                    self.idf_values[token] = math.log((1 + total_docs) / (1 + count)) + 1.0
                else:
                    self.idf_values[token] = math.log(total_docs / count)

    """
    PURPOSE: The non-zero (column, value) pairs for one document.
    WHY SEPARATE FROM transform_one: a document touches a few dozen of tens of
    thousands of columns. Training over 10k rows through the dense form costs
    gigabytes for numbers that are all zero; this is the same arithmetic
    without materialising them.
    """
    def transform_one_sparse(self, text):
        tokens = self.tokenize(text)

        # Count how many times each token appears in THIS specific sentence
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        total_tokens = len(tokens)
        if not total_tokens:
            return np.zeros(0, dtype=np.int64), np.zeros(0)

        columns, values = [], []
        for token, count in token_counts.items():
            index = self.vocab.get(token)
            if index is None:
                continue
            if self.sublinear_tf:
                tf = 1.0 + math.log(count)
            else:
                # TF (Term Frequency) = (Times word appears in sentence) / (Total words in sentence)
                tf = count / total_tokens
            columns.append(index)
            values.append(tf * self.idf_values.get(token, 0.0))

        return (np.array(columns, dtype=np.int64),
                np.array(values, dtype=np.float64))

    """
    PURPOSE: Transforms a single sentence into a numerical array (vector).
    """
    def transform_one(self, text):
        # Create an array of zeros, exactly the size of our vocabulary
        vector = np.zeros(self.vocab_size)
        columns, values = self.transform_one_sparse(text)
        vector[columns] = values
        return vector

    """
    PURPOSE: Transforms a list of sentences into a 2D matrix (used during training).
    """
    def transform(self, documents):
        return np.array([self.transform_one(doc) for doc in documents])

    """
    PURPOSE: Transforms a list of sentences into the sparse triples
    (row indices, column indices, values) plus the matrix shape.

    WHY: this is what keeps training cheap and lets the sweep in
    docs/ML_MODEL_INVESTIGATION.md try representations the dense path could not
    hold. The shipped word 1-2gram model is 29,205 columns, so its dense
    training matrix would be 11,553 x 29,205 float64 = 2.7 GB; the sparse form
    is a few megabytes, because each claim touches about 35 columns. The
    word + character representations the sweep compares against are three times
    wider again.
    """
    def transform_sparse(self, documents, l2_normalize=True):
        documents = list(documents)
        rows, columns, values = [], [], []
        for row, doc in enumerate(documents):
            column, value = self.transform_one_sparse(doc)
            if l2_normalize:
                norm = np.linalg.norm(value)
                if norm > 0:
                    value = value / norm
            rows.append(np.full(len(column), row, dtype=np.int64))
            columns.append(column)
            values.append(value)

        empty_int = np.zeros(0, dtype=np.int64)
        return SparseMatrix(
            rows=np.concatenate(rows) if rows else empty_int,
            columns=np.concatenate(columns) if columns else empty_int,
            values=np.concatenate(values) if values else np.zeros(0),
            n_rows=len(documents),
            n_columns=self.vocab_size,
        )


class SparseMatrix:
    """The three arrays a sparse row-major matrix needs, and the two products
    a linear model needs from it. Deliberately not scipy: the ml-service ships
    without scikit-learn or scipy and the README says so."""

    __slots__ = ("rows", "columns", "values", "n_rows", "n_columns")

    def __init__(self, rows, columns, values, n_rows, n_columns):
        self.rows = rows
        self.columns = columns
        self.values = values
        self.n_rows = n_rows
        self.n_columns = n_columns

    @property
    def shape(self):
        return (self.n_rows, self.n_columns)

    def dot(self, weights):
        """X @ w -- one scalar per row."""
        return np.bincount(
            self.rows,
            weights=self.values * weights[self.columns],
            minlength=self.n_rows,
        )

    def transpose_dot(self, per_row):
        """X.T @ g -- one scalar per feature, i.e. every feature's gradient."""
        return np.bincount(
            self.columns,
            weights=self.values * per_row[self.rows],
            minlength=self.n_columns,
        )

    def to_dense(self):
        dense = np.zeros((self.n_rows, self.n_columns))
        dense[self.rows, self.columns] = self.values
        return dense


# ---------------------------------------------------------------------------
# LOCAL TESTING / DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    columns = [
        "id", "label", "statement", "subject", "speaker",
        "job", "state", "party",
        "barely_true", "false", "half_true", "mostly_true", "pants_fire",
        "context"
    ]

    # read_liar(), not a bare read_csv: the default parser treats `"` as a
    # quote character and merges rows. See binary_truth_mlp.read_liar.
    from binary_truth_mlp import load_split

    df = load_split("train")

    vectorizer = TFIDFVectorizer()

    print("Building vocab and IDF scores...")
    vectorizer.build_vocab(df["statement"])
    print("Vocab size:", vectorizer.vocab_size)

    # Test it on one statement
    test = "Hillary Clinton agrees with John McCain on health care"
    vec = vectorizer.transform_one(test)

    print("\nVector shape:", vec.shape)
    print("Non-zero slots:", np.count_nonzero(vec))
    print("\nTop scoring tokens in this statement:")
    
    # Sort the vector to find the highest TF-IDF scores
    top_indices = np.argsort(vec)[::-1][:10]
    index_to_token = {v: k for k, v in vectorizer.vocab.items()}
    for idx in top_indices:
        if vec[idx] > 0:
            print(f"  {index_to_token[idx]:<20} score: {vec[idx]:.4f}")
