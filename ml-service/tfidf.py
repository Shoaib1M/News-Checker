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

class TFIDFVectorizer:
    def __init__(self, ngram_range=(1, 2), min_df=2):
        self.vocab = {}          # Maps a token to its column index in the final vector
        self.idf_values = {}     # Stores the calculated IDF score for each token
        self.vocab_size = 0
        
        # ngram_range=(1, 2) means we look at single words (unigrams) AND pairs of words (bigrams).
        # Example for "fake news": Unigrams: ["fake", "news"]. Bigrams: ["fake news"].
        self.ngram_range = ngram_range
        
        # min_df (Minimum Document Frequency): Ignore words that appear in fewer than 2 documents.
        # This filters out extremely rare words or typos to keep the vocabulary size manageable.
        self.min_df = min_df

    """
    PURPOSE: Standardizes the text.
    """
    def clean(self, text):
        import re
        text = text.lower()
        # Keep letters, numbers, and spaces. Replace everything else with a space.
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Collapse multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

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
            words = self.clean(doc)
            tokens = set(self.get_ngrams(words)) # Use set() so we only count a word once per document
            for token in tokens:
                doc_frequency[token] = doc_frequency.get(token, 0) + 1

        # Step 2: Calculate the IDF score for tokens that meet the minimum frequency
        for token, count in doc_frequency.items():
            if count >= self.min_df:
                # Assign this token a permanent index/column in our vectors
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1
                
                # Formula for IDF: log( Total Documents / Documents containing word )
                self.idf_values[token] = math.log(total_docs / count)

    """
    PURPOSE: Transforms a single sentence into a numerical array (vector).
    """
    def transform_one(self, text):
        words = self.clean(text)
        tokens = self.get_ngrams(words)
        
        # Create an array of zeros, exactly the size of our vocabulary
        vector = np.zeros(self.vocab_size)

        # Count how many times each token appears in THIS specific sentence
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        total_tokens = len(tokens)
        
        # Calculate TF-IDF
        for token, count in token_counts.items():
            if token in self.vocab:
                # TF (Term Frequency) = (Times word appears in sentence) / (Total words in sentence)
                tf = count / total_tokens
                # Get the pre-calculated IDF score
                idf = self.idf_values.get(token, 0)
                
                # Combine them (TF * IDF) and place the result in the correct slot in the array
                index = self.vocab[token]
                vector[index] = tf * idf

        return vector

    """
    PURPOSE: Transforms a list of sentences into a 2D matrix (used during training).
    """
    def transform(self, documents):
        return np.array([self.transform_one(doc) for doc in documents])


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

    df = pd.read_csv('../data/train.tsv', sep='\t', names=columns)

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
