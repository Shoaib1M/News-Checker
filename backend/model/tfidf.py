import numpy as np
import math

class TFIDFVectorizer:
    def __init__(self):
        self.vocab = {}          
        self.idf_values = {}     
        self.vocab_size = 0

    def clean(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def build_vocab(self, documents):
        # Step 1 — build vocabulary (same as tokenizer)
        for doc in documents:
            words = self.clean(doc)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1

        total_docs = len(documents)

        doc_frequency = {}
        for doc in documents:
            words = set(self.clean(doc))  
            for word in words:
                doc_frequency[word] = doc_frequency.get(word, 0) + 1


        for word, count in doc_frequency.items():
            self.idf_values[word] = math.log(total_docs / count)

    def transform_one(self, text):
        words = self.clean(text)
        vector = np.zeros(self.vocab_size)

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        total_words = len(words)
        for word, count in word_counts.items():
            if word in self.vocab:
                tf = count / total_words
                idf = self.idf_values.get(word, 0)
                index = self.vocab[word]
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
    test = "Hillary Clinton agrees with John McCain"
    vec = vectorizer.transform_one(test)

    print("\nVector shape:", vec.shape)
    print("Non-zero slots:", np.count_nonzero(vec))
    print("\nTop scoring words in this statement:")
    top_indices = np.argsort(vec)[::-1][:6]
    index_to_word = {v: k for k, v in vectorizer.vocab.items()}
    for idx in top_indices:
        print(f"  {index_to_word[idx]:<15} score: {vec[idx]:.4f}")