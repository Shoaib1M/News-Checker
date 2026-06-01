import numpy as np
import math

class TFIDFVectorizer:
    def __init__(self, ngram_range=(1, 2), min_df=2):
        self.vocab = {}          
        self.idf_values = {}     
        self.vocab_size = 0
        self.ngram_range = ngram_range
        self.min_df = min_df

    def clean(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    def get_ngrams(self, words):
        tokens = []
        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            if len(words) < n:
                continue

            for index in range(len(words) - n + 1):
                tokens.append(" ".join(words[index:index + n]))

        return tokens

    def build_vocab(self, documents):
        # Step 1 — build vocabulary (same as tokenizer)
        total_docs = len(documents)

        doc_frequency = {}
        for doc in documents:
            words = self.clean(doc)
            tokens = set(self.get_ngrams(words))
            for token in tokens:
                doc_frequency[token] = doc_frequency.get(token, 0) + 1


        for token, count in doc_frequency.items():
            if count >= self.min_df:
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1
                self.idf_values[token] = math.log(total_docs / count)

    def transform_one(self, text):
        words = self.clean(text)
        tokens = self.get_ngrams(words)
        vector = np.zeros(self.vocab_size)

        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        total_tokens = len(tokens)
        for token, count in token_counts.items():
            if token in self.vocab:
                tf = count / total_tokens
                idf = self.idf_values.get(token, 0)
                index = self.vocab[token]
                vector[index] = tf * idf

        return vector

    def transform(self, documents):
        return np.array([self.transform_one(doc) for doc in documents])


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

    # transform one statement and inspect it
    test = "Hillary Clinton agrees with John McCain on health care"
    vec = vectorizer.transform_one(test)

    print("\nVector shape:", vec.shape)
    print("Non-zero slots:", np.count_nonzero(vec))
    print("\nTop scoring tokens in this statement:")
    top_indices = np.argsort(vec)[::-1][:10]
    index_to_token = {v: k for k, v in vectorizer.vocab.items()}
    for idx in top_indices:
        print(f"  {index_to_token[idx]:<20} score: {vec[idx]:.4f}")
