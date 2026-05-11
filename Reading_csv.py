import pandas as pd


columns = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire",
    "context"
]

df = pd.read_csv('test.tsv', sep='\t', names=columns)

print(df.head())


