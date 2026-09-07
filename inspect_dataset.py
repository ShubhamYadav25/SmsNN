import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

df = pd.read_csv(
    "SMSSpamCollection",

    # The dataset uses TAB between the label and message.
    # Example:
    # ham<TAB>Hey, how are you?
    sep="\t",

    # The original file does NOT contain column names.
    # Therefore pandas should not use the first row as a header.
    header=None,

    # We create our own meaningful column names.
    names=["label", "message"]
)


# ============================================================
# 2. LOOK AT THE FIRST FEW ROWS
# ============================================================

print("\n========== FIRST 5 ROWS ==========")

# head() shows the first 5 rows.
# This lets us visually understand what the data looks like.
print(df.head())


# ============================================================
# 3. LOOK AT THE LAST FEW ROWS
# ============================================================

print("\n========== LAST 5 ROWS ==========")

# tail() shows the last 5 rows.
# Useful for checking whether the file was loaded completely.
print(df.tail())


# ============================================================
# 4. HOW MANY ROWS AND COLUMNS?
# ============================================================

print("\n========== DATASET SHAPE ==========")

# shape returns:
#
#     (number_of_rows, number_of_columns)
#
# For example:
#
#     (5574, 2)
#
# means:
#     5574 messages
#     2 columns
#
print("Shape:", df.shape)

# We can also access them separately.
rows, columns = df.shape

print("Number of rows:", rows)
print("Number of columns:", columns)


# ============================================================
# 5. WHAT ARE OUR COLUMNS?
# ============================================================

print("\n========== COLUMNS ==========")

# columns tells us the names of all columns.
print(df.columns)


# ============================================================
# 6. WHAT TYPE OF DATA IS EACH COLUMN?
# ============================================================

print("\n========== DATA TYPES ==========")

# dtypes tells us what type pandas thinks each column contains.
#
# label   -> usually object/string
# message -> usually object/string
#
# This is important because both are currently TEXT.
print(df.dtypes)


# ============================================================
# 7. HOW MANY EXAMPLES BELONG TO EACH CLASS?
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

# value_counts() counts how many times each unique label appears.
#
# Example:
#
# ham     4827
# spam     747
#
# This tells us how many examples we have
# in each class.
label_counts = df["label"].value_counts()

print(label_counts)


# ============================================================
# 8. CLASS DISTRIBUTION AS PERCENTAGE
# ============================================================

print("\n========== CLASS DISTRIBUTION (%) ==========")

# value_counts(normalize=True) gives proportions instead of counts.
#
# Example:
#
# ham     0.866
# spam    0.134
#
# Multiply by 100 to convert proportions into percentages.
label_percentages = df["label"].value_counts(normalize=True) * 100

print(label_percentages.round(2))


# ============================================================
# 9. TOTAL NUMBER OF MESSAGES
# ============================================================

print("\n========== TOTAL MESSAGES ==========")

# len(df) gives the number of rows.
#
# Every row represents one message.
print("Total messages:", len(df))


# ============================================================
# 10. COUNT HAM AND SPAM SEPARATELY
# ============================================================

print("\n========== HAM / SPAM COUNTS ==========")

# Count only rows where label == "ham"
ham_count = (df["label"] == "ham").sum()

# Count only rows where label == "spam"
spam_count = (df["label"] == "spam").sum()

print("Ham messages:", ham_count)
print("Spam messages:", spam_count)


# ============================================================
# 11. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

# isnull() checks every cell.
#
# True  -> value is missing
# False -> value exists
#
# sum() then counts the True values.
print(df.isnull().sum())


# ============================================================
# 12. CHECK DUPLICATE ROWS
# ============================================================

print("\n========== DUPLICATES ==========")

# duplicated() checks whether a complete row
# has appeared before.
#
# keep=False means:
# mark ALL copies of duplicated rows as True.
#
# sum() counts them.
duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ============================================================
# 13. SEE SOME DUPLICATES
# ============================================================

print("\n========== DUPLICATE EXAMPLES ==========")

# Select every row that belongs to a duplicated group.
duplicates = df[df.duplicated(keep=False)]

# Display first 20 duplicate-related rows.
print(duplicates.head(20))

# ============================================================
# 15. BASIC MESSAGE-LENGTH STATISTICS
# ============================================================

print("\n========== MESSAGE LENGTH STATISTICS ==========")

# describe() gives common statistical information:
#
# count  -> number of messages
# mean   -> average message length
# std    -> how spread out the lengths are
# min    -> shortest message
# 25%    -> first quartile
# 50%    -> median
# 75%    -> third quartile
# max    -> longest message
#
print(df["message_length"].describe())

# ============================================================
# 16. MESSAGE LENGTH BY CLASS
# ============================================================

print("\n========== LENGTH BY CLASS ==========")

# groupby("label") means:
#
# "Separate the dataset into ham and spam groups."
#
# Then we calculate statistics for message_length
# independently for each group.
length_by_class = df.groupby("label")["message_length"].describe()

print(length_by_class)

# ============================================================
# 17. WORD COUNT
# ============================================================

# split() separates the message into pieces based on whitespace.
#
# Example:
#
# "Hey call me later"
#
# becomes approximately:
#
# ["Hey", "call", "me", "later"]
#
# len() then counts those pieces.
df["word_count"] = df["message"].str.split().str.len()

print("\n========== WORD COUNT ==========")
print(df["word_count"].head())

# ============================================================
# 18. WORD COUNT BY CLASS
# ============================================================

print("\n========== WORD COUNT BY CLASS ==========")

word_count_by_class = df.groupby("label")["word_count"].describe()

print(word_count_by_class)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

# Count how many messages belong to each class.
class_counts = df["label"].value_counts()

# Create a bar chart.
class_counts.plot(kind="bar")

# Give the graph a title.
plt.title("Ham vs Spam Messages")

# Label the x-axis.
plt.xlabel("Class")

# Label the y-axis.
plt.ylabel("Number of Messages")

# Make the x-axis labels easier to read.
plt.xticks(rotation=0)

# Display the graph.
plt.show()

df["message_length"].plot(
    kind="hist",
    bins=50
)

plt.title("Message Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Messages")

plt.show()

# ============================================================
# MESSAGE LENGTH: HAM VS SPAM
# ============================================================

# Select only ham messages.
ham_lengths = df[df["label"] == "ham"]["message_length"]

# Select only spam messages.
spam_lengths = df[df["label"] == "spam"]["message_length"]

# Draw the ham distribution.
ham_lengths.plot(
    kind="hist",
    bins=50,
    alpha=0.6,
    label="ham"
)

# Draw the spam distribution on the same graph.
spam_lengths.plot(
    kind="hist",
    bins=50,
    alpha=0.6,
    label="spam"
)

plt.title("Message Length: Ham vs Spam")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Messages")

# Display the legend so we know which distribution is which.
plt.legend()

plt.show()