import pandas as pd


columns = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire",
    "context"
]                     

df = pd.read_csv('../data/train.tsv', sep='\t', names=columns)

print(df.head())

print("Shape of the dataset:",df.shape)

print('Number of labels:', df['label'].value_counts())

print('Sample statement:', df['statement'][0])


print("\nMissing values:")
print(df.isnull().sum())