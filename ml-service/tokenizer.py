"""
FILE PURPOSE:
This file defines a Tokenizer class. 
A Tokenizer takes raw human text (like "Hello world!") and converts it into a list of numbers 
that a machine learning model can understand.

FLOW:
1. `clean()`: Removes punctuation and makes everything lowercase.
2. `build_vocab()`: Reads a large dataset of text and assigns a unique ID to every word it sees.
3. `tokenize()`: Takes a new sentence and replaces every word with its assigned ID.

USED BY:
- Historically used by older, simpler models in this project.
- Currently, the main ML model uses `TFIDFVectorizer` (in tfidf.py) instead, but this file 
  remains as a foundational building block for understanding NLP (Natural Language Processing).
"""

import re

class Tokenizer:
    def __init__(self):
        # Maps a word to its unique integer ID (e.g., {"apple": 0, "banana": 1})
        self.word_to_id = {}
        # Maps the integer ID back to the word (e.g., {0: "apple", 1: "banana"})
        self.id_to_word = {}
        # Keeps track of how many unique words we've seen
        self.vocab_size = 0

    """
    PURPOSE:
    Standardizes the text so the model doesn't get confused.

    INPUT:
    Raw string (e.g., "Hello, World!!!")

    OUTPUT:
    List of clean lowercase words (e.g., ["hello", "world"])

    WHY THIS EXISTS:
    To a computer, "Apple", "apple", and "apple!" look like completely different words. 
    Cleaning ensures they are all treated as the same word.
    """
    def clean(self, text):
        # Step 1: Make everything lowercase
        text = text.lower()
        # Step 2: Remove anything that is NOT a word character (\w) or a space (\s) using regular expressions (Regex)
        text = re.sub(r'[^\w\s]', '', text)
        # Step 3: Split the string by spaces into a list of words
        return text.split()
    
    """
    PURPOSE:
    Learns the vocabulary from a training dataset.

    INPUT:
    texts: A list of sentences/strings.

    WHY THIS EXISTS:
    Before we can convert words to numbers, we need a dictionary that tells us which number belongs to which word.
    """
    def build_vocab(self, texts):
        for text in texts:
            # Clean each sentence into a list of words
            for word in self.clean(text):
                # If we've never seen this word before...
                if word not in self.word_to_id:
                    # Give it the next available ID
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word
                    # Increment the counter for the next word
                    self.vocab_size += 1

    """
    PURPOSE:
    Converts a real sentence into a list of numbers using the learned vocabulary.

    INPUT:
    text: A raw string

    OUTPUT:
    A list of integers.

    WHY THIS EXISTS:
    Neural networks only understand numbers.
    """
    def tokenize(self, text):
        tokens = self.clean(text)
        result = []
        for token in tokens:
            # If we know the word, append its ID.
            # If we've never seen this word before (Out of Vocabulary), we append -1.
            result.append(self.word_to_id.get(token, -1))
        return result
    

# ---------------------------------------------------------------------------
# LOCAL TESTING / DEMO
# If you run `python tokenizer.py` directly, this block will execute to show you how it works.
# ---------------------------------------------------------------------------
if __name__== '__main__':
    import pandas as pd

    # Define the columns of the training dataset
    columns = [
        "id", "label", "statement", "subject", "speaker",
        "job", "state", "party",
        "barely_true", "false", "half_true", "mostly_true", "pants_fire",
        "context"
    ]

    # Load the training data
    df = pd.read_csv('../data/train.tsv', sep='\t', names=columns)

    tokenizer = Tokenizer()

    # Build vocabulary from all statements in the dataset
    print("Building vocabulary...")
    tokenizer.build_vocab(df["statement"])

    print("Vocab size:", tokenizer.vocab_size)

    # Test it on one sentence to see the output
    test = "Hillary Clinton agrees with John McCain"
    print("\nOriginal:", test)
    print("Tokens:  ", tokenizer.tokenize(test))
    print("Words:   ", tokenizer.clean(test))