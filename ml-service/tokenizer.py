import re


class Tokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def clean(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def build_vocab(self, texts):
        for text in texts:
            for word in self.clean(text):
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word
                    self.vocab_size += 1
    def tokenize(self, text):
        tokens = self.clean(text)
        result = []
        for token in tokens:
            result.append(self.word_to_id.get(token,-1))
        return result
    


if __name__== '__main__':
    import pandas as pd

    columns = [
        "id", "label", "statement", "subject", "speaker",
        "job", "state", "party",
        "barely_true", "false", "half_true", "mostly_true", "pants_fire",
        "context"
    ]

    df = pd.read_csv('../data/train.tsv', sep='\t', names=columns)

    tokenizer = Tokenizer()

    # build vocabulary from all statements
    tokenizer.build_vocab(df["statement"])

    print("Vocab size:", tokenizer.vocab_size)

    # test it on one sentence
    test = "Hillary Clinton agrees with John McCain"
    print("\nOriginal:", test)
    print("Tokens:  ", tokenizer.tokenize(test))
    print("Words:   ", tokenizer.clean(test))