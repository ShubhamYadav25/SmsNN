# ============================================================
# SCAM / SPAM DETECTOR
# PHASE 3 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
#
# What this program does:
#
# 1. Loads the SMS Spam dataset
# 2. Inspects the raw data
# 3. Checks dataset size
# 4. Checks classes
# 5. Checks class imbalance
# 6. Checks missing values
# 7. Checks duplicate messages
# 8. Calculates message length
# 9. Calculates word count
# 10. Calculates digit count
# 11. Detects URLs
# 12. Calculates uppercase characters
# 13. Calculates ! and ?
# 14. Compares features between ham and spam
# 15. Creates visualizations
#
# IMPORTANT:
# We are NOT training a neural network yet.
# We are only trying to understand our data.
# ============================================================


import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "SMSSpamCollection",

    # The original dataset uses a TAB between
    # the label and the message.
    #
    # Example:
    #
    # ham<TAB>Hey, how are you?
    #
    sep="\t",

    # The original file doesn't contain column names.
    header=None,

    # We create our own column names.
    names=["label", "message"]
)


# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================

print("\n")
print("=" * 60)
print("1. FIRST 5 ROWS")
print("=" * 60)

# head() displays the first 5 rows.
# This lets us see what the dataset actually looks like.
print(df.head())


print("\n")
print("=" * 60)
print("2. LAST 5 ROWS")
print("=" * 60)

# tail() displays the last 5 rows.
print(df.tail())


# ============================================================
# 3. DATASET SHAPE
# ============================================================

print("\n")
print("=" * 60)
print("3. DATASET SHAPE")
print("=" * 60)

# shape returns:
#
#     (rows, columns)
#
# For example:
#
#     (5574, 2)
#
# means:
#
#     5574 rows
#     2 columns
#
print("Shape:", df.shape)

rows, columns = df.shape

print("Number of rows:", rows)
print("Number of columns:", columns)


# ============================================================
# 4. COLUMN NAMES
# ============================================================

print("\n")
print("=" * 60)
print("4. COLUMN NAMES")
print("=" * 60)

print(df.columns)


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n")
print("=" * 60)
print("5. DATA TYPES")
print("=" * 60)

# dtypes tells us what type of data each column contains.
#
# At this stage both label and message are text-like values.
#
print(df.dtypes)


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("6. CLASS DISTRIBUTION")
print("=" * 60)

# value_counts() counts how many times
# each unique label occurs.
#
# Example:
#
# ham     4827
# spam     747
#
label_counts = df["label"].value_counts()

print(label_counts)


# ============================================================
# 7. CLASS DISTRIBUTION AS PERCENTAGE
# ============================================================

print("\n")
print("=" * 60)
print("7. CLASS DISTRIBUTION (%)")
print("=" * 60)

# normalize=True changes counts into proportions.
#
# Example:
#
# ham     0.866
# spam    0.134
#
# Multiplying by 100 converts these into percentages.
#
label_percentages = (
    df["label"]
    .value_counts(normalize=True)
    * 100
)

print(label_percentages.round(2))


# ============================================================
# 8. HAM AND SPAM COUNTS
# ============================================================

print("\n")
print("=" * 60)
print("8. HAM / SPAM COUNTS")
print("=" * 60)

# Create a Boolean condition:
#
# df["label"] == "ham"
#
# produces:
#
# True
# False
# True
# ...
#
# sum() counts the True values.
#
ham_count = (df["label"] == "ham").sum()
spam_count = (df["label"] == "spam").sum()

print("Ham messages :", ham_count)
print("Spam messages:", spam_count)

print("Total messages:", ham_count + spam_count)


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n")
print("=" * 60)
print("9. MISSING VALUES")
print("=" * 60)

# isnull() checks whether values are missing.
#
# True  -> missing
# False -> present
#
# sum() counts how many missing values exist.
#
missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# 10. DUPLICATES
# ============================================================

print("\n")
print("=" * 60)
print("10. DUPLICATE ROWS")
print("=" * 60)

# duplicated() checks whether a complete row
# has appeared previously.
#
# sum() counts the duplicate rows.
#
duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ============================================================
# 11. DISPLAY DUPLICATE EXAMPLES
# ============================================================

print("\n")
print("=" * 60)
print("11. DUPLICATE EXAMPLES")
print("=" * 60)

# keep=False marks ALL rows belonging to a
# duplicated group as True.
#
duplicates = df[df.duplicated(keep=False)]

print(duplicates.head(20))


# ============================================================
# 12. MESSAGE LENGTH
# ============================================================

print("\n")
print("=" * 60)
print("12. MESSAGE LENGTH")
print("=" * 60)

# str.len() counts the number of characters
# in every message.
#
# Example:
#
# "Hello"
#    ↓
# 5
#
df["message_length"] = df["message"].str.len()

print(
    df[
        ["message", "message_length"]
    ].head(10)
)


# ============================================================
# 13. MESSAGE LENGTH STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("13. MESSAGE LENGTH STATISTICS")
print("=" * 60)

# describe() calculates common statistics:
#
# count -> number of messages
# mean  -> average
# std   -> standard deviation
# min   -> smallest value
# 25%   -> first quartile
# 50%   -> median
# 75%   -> third quartile
# max   -> largest value
#
print(df["message_length"].describe())


# ============================================================
# 14. MESSAGE LENGTH BY CLASS
# ============================================================

print("\n")
print("=" * 60)
print("14. MESSAGE LENGTH BY CLASS")
print("=" * 60)

# groupby("label") separates the dataset into:
#
# ham
# spam
#
# Then describe() calculates statistics
# separately for each class.
#
length_by_class = (
    df
    .groupby("label")["message_length"]
    .describe()
)

print(length_by_class)


# ============================================================
# 15. WORD COUNT
# ============================================================

print("\n")
print("=" * 60)
print("15. WORD COUNT")
print("=" * 60)

# split() separates a sentence into pieces
# based on whitespace.
#
# Example:
#
# "Hey call me later"
#
# becomes approximately:
#
# ["Hey", "call", "me", "later"]
#
# str.len() then counts those words.
#
df["word_count"] = (
    df["message"]
    .str.split()
    .str.len()
)

print(
    df[
        ["message", "word_count"]
    ].head(10)
)


# ============================================================
# 16. WORD COUNT BY CLASS
# ============================================================

print("\n")
print("=" * 60)
print("16. WORD COUNT BY CLASS")
print("=" * 60)

word_count_by_class = (
    df
    .groupby("label")["word_count"]
    .describe()
)

print(word_count_by_class)


# ============================================================
# 17. DIGIT COUNT
# ============================================================

print("\n")
print("=" * 60)
print("17. DIGIT COUNT")
print("=" * 60)

# \d means a numeric digit.
#
# Example:
#
# "Win £5000 now"
#
# contains:
#
# 5
# 0
# 0
# 0
#
# Therefore digit_count = 4
#
df["digit_count"] = (
    df["message"]
    .str.count(r"\d")
)

print(
    df[
        ["message", "digit_count"]
    ].head(10)
)


# ============================================================
# 18. DIGIT COUNT BY CLASS
# ============================================================

print("\n")
print("=" * 60)
print("18. DIGIT COUNT BY CLASS")
print("=" * 60)

digit_by_class = (
    df
    .groupby("label")["digit_count"]
    .describe()
)

print(digit_by_class)


# ============================================================
# 19. URL DETECTION
# ============================================================

print("\n")
print("=" * 60)
print("19. URL DETECTION")
print("=" * 60)

# We search for common URL patterns:
#
# http
# https
# www.
#
# case=False means:
#
# HTTP
# http
# Http
#
# are all considered matches.
#
df["contains_url"] = (
    df["message"]
    .str.contains(
        r"http|www\.",
        case=False,
        regex=True,
        na=False
    )
)

print(
    df[
        ["message", "contains_url"]
    ].head(10)
)


# ============================================================
# 20. URL COUNTS
# ============================================================

print("\n")
print("=" * 60)
print("20. URL PRESENCE")
print("=" * 60)

# Count True and False.
#
# True  -> URL detected
# False -> URL not detected
#
print(df["contains_url"].value_counts())


# ============================================================
# 21. URL PRESENCE BY CLASS
# ============================================================

print("\n")
print("=" * 60)
print("21. URL PRESENCE BY CLASS")
print("=" * 60)

# crosstab creates a frequency table.
#
# Rows:
#     ham
#     spam
#
# Columns:
#     False
#     True
#
url_by_class = pd.crosstab(
    df["label"],
    df["contains_url"]
)

print(url_by_class)


# ============================================================
# 22. UPPERCASE CHARACTER COUNT
# ============================================================

print("\n")
print("=" * 60)
print("22. UPPERCASE CHARACTERS")
print("=" * 60)

# [A-Z] matches uppercase English letters.
#
# Example:
#
# "HELLO"
#
# contains 5 uppercase characters.
#
df["uppercase_count"] = (
    df["message"]
    .str.count(r"[A-Z]")
)

print(
    df[
        ["message", "uppercase_count"]
    ].head(10)
)


# ============================================================
# 23. EXCLAMATION MARKS
# ============================================================

print("\n")
print("=" * 60)
print("23. EXCLAMATION MARKS")
print("=" * 60)

# Count how many ! characters appear.
#
# "WIN NOW!!!"
#
# gives:
#
# 3
#
df["exclamation_count"] = (
    df["message"]
    .str.count("!")
)

print(
    df[
        ["message", "exclamation_count"]
    ].head(10)
)


# ============================================================
# 24. QUESTION MARKS
# ============================================================

print("\n")
print("=" * 60)
print("24. QUESTION MARKS")
print("=" * 60)

# Count question marks.
#
# ? has special meaning in regular expressions,
# so we escape it as:
#
# \?
#
df["question_count"] = (
    df["message"]
    .str.count(r"\?")
)

print(
    df[
        ["message", "question_count"]
    ].head(10)
)


# ============================================================
# 25. SUMMARY OF NUMERICAL FEATURES
# ============================================================

print("\n")
print("=" * 60)
print("25. FEATURE STATISTICS")
print("=" * 60)

# These are numerical characteristics we have
# extracted from the messages.
#
feature_columns = [
    "message_length",
    "word_count",
    "digit_count",
    "uppercase_count",
    "exclamation_count",
    "question_count"
]

# describe() gives statistics for every feature.
print(
    df[feature_columns].describe()
)


# ============================================================
# 26. FEATURE AVERAGES BY CLASS
# ============================================================

print("\n")
print("=" * 60)
print("26. FEATURE AVERAGES: HAM VS SPAM")
print("=" * 60)

# groupby("label") separates ham and spam.
#
# mean() calculates the average of each
# numerical feature for each class.
#
feature_means = (
    df
    .groupby("label")[feature_columns]
    .mean()
)

print(feature_means.round(2))


# ============================================================
# 27. VISUALIZATION - CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("27. CLASS DISTRIBUTION GRAPH")
print("=" * 60)

class_counts.plot(kind="bar")

plt.title("Ham vs Spam Messages")
plt.xlabel("Class")
plt.ylabel("Number of Messages")

# Keep labels horizontal.
plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 28. VISUALIZATION - MESSAGE LENGTH
# ============================================================

print("\n")
print("=" * 60)
print("28. MESSAGE LENGTH DISTRIBUTION")
print("=" * 60)

df["message_length"].plot(
    kind="hist",
    bins=50
)

plt.title("Message Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Messages")

plt.tight_layout()

plt.show()


# ============================================================
# 29. VISUALIZATION - HAM VS SPAM LENGTH
# ============================================================

print("\n")
print("=" * 60)
print("29. MESSAGE LENGTH: HAM VS SPAM")
print("=" * 60)

# Extract message lengths belonging to ham.
ham_lengths = df[
    df["label"] == "ham"
]["message_length"]

# Extract message lengths belonging to spam.
spam_lengths = df[
    df["label"] == "spam"
]["message_length"]

# Draw ham distribution.
ham_lengths.plot(
    kind="hist",
    bins=50,
    alpha=0.6,
    label="ham"
)

# Draw spam distribution.
spam_lengths.plot(
    kind="hist",
    bins=50,
    alpha=0.6,
    label="spam"
)

plt.title("Message Length: Ham vs Spam")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Messages")

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 30. VISUALIZATION - WORD COUNT
# ============================================================

print("\n")
print("=" * 60)
print("30. WORD COUNT DISTRIBUTION")
print("=" * 60)

df["word_count"].plot(
    kind="hist",
    bins=40
)

plt.title("Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Messages")

plt.tight_layout()

plt.show()


# ============================================================
# 31. VISUALIZATION - HAM VS SPAM WORD COUNT
# ============================================================

print("\n")
print("=" * 60)
print("31. WORD COUNT: HAM VS SPAM")
print("=" * 60)

ham_words = df[
    df["label"] == "ham"
]["word_count"]

spam_words = df[
    df["label"] == "spam"
]["word_count"]

ham_words.plot(
    kind="hist",
    bins=40,
    alpha=0.6,
    label="ham"
)

spam_words.plot(
    kind="hist",
    bins=40,
    alpha=0.6,
    label="spam"
)

plt.title("Word Count: Ham vs Spam")
plt.xlabel("Number of Words")
plt.ylabel("Number of Messages")

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 32. FINAL DATASET PREVIEW
# ============================================================

print("\n")
print("=" * 60)
print("32. FINAL DATASET PREVIEW")
print("=" * 60)

# Now we can see our original columns plus
# the numerical characteristics we created.
#
print(df.head())


# ============================================================
# 33. FINAL COLUMN LIST
# ============================================================

print("\n")
print("=" * 60)
print("33. FINAL COLUMNS")
print("=" * 60)

print(df.columns.tolist())


# ============================================================
# END OF EDA
# ============================================================

print("\n")
print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print("""
We have now investigated:

✓ Dataset size
✓ Columns
✓ Data types
✓ Class distribution
✓ Class imbalance
✓ Missing values
✓ Duplicate rows
✓ Message length
✓ Word count
✓ Digit count
✓ URLs
✓ Uppercase characters
✓ Exclamation marks
✓ Question marks
✓ Feature statistics
✓ Ham vs spam statistics
✓ Basic visualizations

Next:
TEXT → NUMBERS

We will learn:
tokens
vocabulary
word frequency
Bag of Words
TF-IDF
""")
