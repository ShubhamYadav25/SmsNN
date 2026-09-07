"""
pandas for data handling
Pandas gives us a convenient table structure called a DataFrame
"""
import pandas as pd

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(df.head())

"""
``
  label                                            message
0   ham    Go until jurong point, crazy.. Available only...
1   ham    Ok lar... Joking wif u oni...
2  spam    Free entry in 2 a wkly comp to win FA Cup...
3   ham    U dun say so early hor...
4   ham    Nah I don't think he goes to usf, he lives...
"""
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nLabels:")
print(df["label"].value_counts())

# If approximately 87% of the dataset is ham, this terrible model could still achieve around: 87% accuracy
"""
That's why later we'll as
Accuracy?
Precision?
Recall?
F1?
Confusion matrix?
"""


print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

duplicates = df[df.duplicated(keep=False)]

print("\nDuplicate examples:")
print(duplicates.head(20))