# ============================================================
# SCAM / SPAM DETECTOR
# PHASE 4 - TEXT -> NUMBERS
# TF-IDF FROM SCRATCH
# ============================================================
#
# In this file we will learn:
#
# 1. Why Bag of Words is useful
# 2. Why Bag of Words has a limitation
# 3. Term Frequency (TF)
# 4. Document Frequency (DF)
# 5. Inverse Document Frequency (IDF)
# 6. TF-IDF
# 7. Build TF-IDF ourselves
# 8. Convert real SMS messages into vectors
# 9. Understand sparse vectors
#
# IMPORTANT:
#
# We are NOT using:
#
#   sklearn
#   TfidfVectorizer
#   neural networks
#
# We are implementing the important ideas ourselves
# so that we understand what is actually happening.
# ============================================================


import math
import pandas as pd


# ============================================================
# PART 1
# WHY DO WE NEED TF-IDF?
# ============================================================

# Our neural network works with numbers.
#
# It cannot directly calculate:
#
#     weight * "money"
#
# Therefore we need:
#
#     text
#       ↓
#     numerical representation
#
#
# We already learned Bag of Words.
#
# Example:
#
# Vocabulary:
#
# ["free", "money", "win", "call"]
#
# Message:
#
# "free money"
#
# Bag of Words:
#
# [1, 1, 0, 0]
#
#
# The 1 means:
#
# "this word appears"
#
# The 0 means:
#
# "this word does not appear"
#
#
# But Bag of Words has a problem:
#
# It treats every word based mainly on its occurrence/count.
#
# Some words are extremely common:
#
#     the
#     to
#     a
#     is
#     you
#
# Other words may be much more specific:
#
#     prize
#     winner
#     claim
#
#
# We would like a representation where:
#
# COMMON WORD
#      ↓
# less informative
#
# RARE / SPECIFIC WORD
#      ↓
# potentially more informative
#
#
# TF-IDF helps us do this.
# ============================================================


# ============================================================
# PART 2
# TERM FREQUENCY (TF)
# ============================================================

# TF answers:
#
#     "How important/frequent is this word
#      INSIDE THIS PARTICULAR MESSAGE?"
#
#
# Example:
#
# Message:
#
#     "free free money"
#
# There are 3 words.
#
# "free" appears 2 times.
#
# Therefore:
#
#     TF(free) = 2 / 3
#
# "money" appears 1 time.
#
#     TF(money) = 1 / 3
#
#
# Formula:
#
#             count(word in document)
# TF = -------------------------------------
#             total words in document
#
#
# ============================================================


def term_frequency(tokens):

    # --------------------------------------------------------
    # Step 1: Count each word
    # --------------------------------------------------------
    #
    # Example:
    #
    # tokens =
    #
    # ["free", "free", "money"]
    #
    # counts becomes:
    #
    # {
    #     "free": 2,
    #     "money": 1
    # }
    #
    counts = {}

    for token in tokens:

        # If this is the first time we see the word,
        # start its count at zero.
        if token not in counts:
            counts[token] = 0

        # Increase the count.
        counts[token] += 1


    # --------------------------------------------------------
    # Step 2: Count total words
    # --------------------------------------------------------

    total_words = len(tokens)


    # --------------------------------------------------------
    # Step 3: Convert counts to frequencies
    # --------------------------------------------------------

    tf = {}

    for word, count in counts.items():

        # Example:
        #
        # count = 2
        # total_words = 3
        #
        # TF = 2 / 3
        #
        tf[word] = count / total_words


    return tf


# ============================================================
# TEST TF
# ============================================================

print("\n")
print("=" * 70)
print("1. TERM FREQUENCY (TF)")
print("=" * 70)

tokens = [
    "free",
    "free",
    "money"
]

tf = term_frequency(tokens)

print("Tokens:")
print(tokens)

print("\nTF:")

for word, value in tf.items():

    print(
        f"{word:10} {value:.4f}"
    )


# Expected idea:
#
# free  -> 0.6667
# money -> 0.3333
#
# Because:
#
# free  = 2 / 3
# money = 1 / 3


# ============================================================
# PART 3
# DOCUMENT FREQUENCY (DF)
# ============================================================

# Now we need to understand another concept.
#
# TF looked INSIDE ONE MESSAGE.
#
# DF looks ACROSS ALL MESSAGES.
#
#
# Suppose we have:
#
# Document 1:
#     "call me tomorrow"
#
# Document 2:
#     "call me now"
#
# Document 3:
#     "win free prize now"
#
#
# The word "call" appears in:
#
# Document 1
# Document 2
#
# Therefore:
#
# DF(call) = 2
#
#
# The word "prize" appears only in:
#
# Document 3
#
# Therefore:
#
# DF(prize) = 1
#
#
# DF answers:
#
# "How many different documents contain this word?"
#
#
# IMPORTANT:
#
# If a word appears 10 times in ONE message,
# it still counts as only ONE document for DF.
#
# Therefore we use:
#
#     set(tokens)
#
# to remove repeated words inside the same document.
# ============================================================


def document_frequency(documents):

    # This will store:
    #
    # word -> number of documents containing the word
    #
    df = {}


    # Look at every document.
    for tokens in documents:

        # We only care whether a word exists
        # in this document.
        #
        # If:
        #
        # ["free", "free", "money"]
        #
        # set() gives:
        #
        # {"free", "money"}
        #
        unique_words = set(tokens)


        # Count each word only once for this document.
        for word in unique_words:

            if word not in df:
                df[word] = 0

            df[word] += 1


    return df


# ============================================================
# TEST DF
# ============================================================

print("\n")
print("=" * 70)
print("2. DOCUMENT FREQUENCY (DF)")
print("=" * 70)


documents = [
    ["call", "me", "tomorrow"],
    ["call", "me", "now"],
    ["win", "free", "prize", "now"]
]


df_counts = document_frequency(documents)

print("Documents:")

for document in documents:
    print(document)


print("\nDocument frequency:")

for word, count in df_counts.items():

    print(
        f"{word:10} {count}"
    )


# Expected idea:
#
# call     -> 2
# me       -> 2
# now      -> 2
# tomorrow -> 1
# win      -> 1
# free     -> 1
# prize    -> 1


# ============================================================
# PART 4
# INVERSE DOCUMENT FREQUENCY (IDF)
# ============================================================

# Now we want to turn DF into a measure of
# how informative a word is.
#
#
# Basic idea:
#
# Word appears in MANY documents
#          ↓
# very common
#          ↓
# less informative
#
#
# Word appears in FEW documents
#          ↓
# more specific
#          ↓
# potentially more informative
#
#
# A simple IDF formula is:
#
#                 N
# IDF(word) = log(-----)
#                 DF
#
#
# Where:
#
# N
# ↓
# total number of documents
#
# DF
# ↓
# number of documents containing the word
#
#
# Example:
#
# N = 4
#
# "prize" appears in 2 documents
#
# IDF(prize):
#
#     log(4 / 2)
#
#     = log(2)
#
#     ≈ 0.693
#
#
# If a word appears in every document:
#
#     DF = N
#
# then:
#
#     log(N / N)
#
#     = log(1)
#
#     = 0
#
#
# Therefore a word appearing everywhere gets
# a very small / zero IDF in this simple formulation.
# ============================================================


def inverse_document_frequency(documents):

    # Total number of documents.
    total_documents = len(documents)

    # First calculate DF.
    df = document_frequency(documents)

    # Store IDF values here.
    idf = {}


    # Calculate IDF for every word.
    for word, document_count in df.items():

        idf[word] = math.log(
            total_documents / document_count
        )


    return idf


# ============================================================
# TEST IDF
# ============================================================

print("\n")
print("=" * 70)
print("3. INVERSE DOCUMENT FREQUENCY (IDF)")
print("=" * 70)


idf = inverse_document_frequency(documents)

print("IDF values:")

for word, value in sorted(idf.items()):

    print(
        f"{word:10} {value:.4f}"
    )


# ============================================================
# PART 5
# TF-IDF
# ============================================================

# Now combine TF and IDF.
#
#
# Formula:
#
#     TF-IDF = TF × IDF
#
#
# TF tells us:
#
#     "How frequently does this word appear
#      in THIS message?"
#
#
# IDF tells us:
#
#     "How uncommon is this word across
#      ALL messages?"
#
#
# Therefore:
#
#     TF × IDF
#
# gives us a value that considers BOTH.
#
#
# A word can have:
#
# HIGH TF
# but
# LOW IDF
#
# because it is common everywhere.
#
#
# Or:
#
# LOW TF
# but
# HIGH IDF
#
# because it is relatively specific.
# ============================================================


def tf_idf(tokens, idf):

    # Calculate TF for this message.
    tf = term_frequency(tokens)

    # Store TF-IDF values.
    result = {}


    # Calculate TF × IDF for every word.
    for word, tf_value in tf.items():

        # Look up the IDF value.
        #
        # If the word isn't known,
        # use 0.
        idf_value = idf.get(word, 0)

        # Multiply TF and IDF.
        result[word] = (
            tf_value * idf_value
        )


    return result


# ============================================================
# TEST TF-IDF
# ============================================================

print("\n")
print("=" * 70)
print("4. TF-IDF")
print("=" * 70)


test_document = [
    "win",
    "free",
    "prize",
    "now"
]


result = tf_idf(
    test_document,
    idf
)


print("Document:")
print(test_document)

print("\nTF-IDF:")

for word, value in result.items():

    print(
        f"{word:10} {value:.4f}"
    )


# ============================================================
# PART 6
# LOAD REAL SMS DATASET
# ============================================================

print("\n")
print("=" * 70)
print("5. REAL SMS DATASET")
print("=" * 70)


df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


print("Number of messages:")
print(len(df))


# ============================================================
# PART 7
# TOKENIZE REAL DATASET
# ============================================================

# We need to turn every message into tokens.
#
# We will keep preprocessing simple for now.
#
# Later we can make preprocessing more sophisticated.
# ============================================================


import string


def tokenize(message):

    # Convert everything to lowercase.
    message = message.lower()

    # Remove punctuation.
    message = message.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Split into words.
    tokens = message.split()

    return tokens


# ============================================================
# TEST TOKENIZER
# ============================================================

print("\n")
print("=" * 70)
print("6. TOKENIZATION ON REAL DATA")
print("=" * 70)


for message in df["message"].head(5):

    print("\nOriginal:")
    print(message)

    print("Tokens:")
    print(tokenize(message))


# ============================================================
# PART 8
# TOKENIZE EVERY DOCUMENT
# ============================================================

# Each SMS becomes one "document".
#
# Therefore:
#
#     1 SMS = 1 document
#
#
# documents will become:
#
# [
#     ["hello", "how", "are", "you"],
#     ["win", "free", "money"],
#     ...
# ]
# ============================================================


documents = []

for message in df["message"]:

    tokens = tokenize(message)

    documents.append(tokens)


print("\n")
print("=" * 70)
print("7. DOCUMENTS")
print("=" * 70)

print("Number of documents:", len(documents))

print("\nFirst document:")
print(documents[0])


# ============================================================
# PART 9
# BUILD VOCABULARY
# ============================================================

# Vocabulary = all unique words across all documents.
#
# Example:
#
# Documents:
#
# ["free", "money"]
# ["free", "prize"]
#
# Vocabulary:
#
# ["free", "money", "prize"]
#
#
# We use a set because it automatically removes duplicates.
# ============================================================


vocabulary = set()

for tokens in documents:

    vocabulary.update(tokens)


print("\n")
print("=" * 70)
print("8. VOCABULARY")
print("=" * 70)

print("Vocabulary size:")
print(len(vocabulary))

print("\nFirst 50 vocabulary words:")

print(
    sorted(vocabulary)[:50]
)


# ============================================================
# PART 10
# CALCULATE REAL IDF
# ============================================================

print("\n")
print("=" * 70)
print("9. CALCULATING IDF FOR REAL DATA")
print("=" * 70)


idf = inverse_document_frequency(
    documents
)


print("Number of IDF values:")
print(len(idf))


# ============================================================
# PART 11
# SEE SOME IDF VALUES
# ============================================================

print("\nSome IDF values:")

for word in list(sorted(idf))[:30]:

    print(
        f"{word:20} {idf[word]:.4f}"
    )


# ============================================================
# PART 12
# CALCULATE TF-IDF FOR ONE REAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("10. TF-IDF FOR ONE REAL MESSAGE")
print("=" * 70)


# Pick the first message.
message = df.iloc[0]["message"]

# Tokenize it.
tokens = tokenize(message)

# Calculate TF-IDF.
vector_values = tf_idf(
    tokens,
    idf
)


print("Original message:")
print(message)

print("\nTokens:")
print(tokens)

print("\nTF-IDF values:")

for word, value in vector_values.items():

    print(
        f"{word:20} {value:.4f}"
    )


# ============================================================
# PART 13
# THE SPARSE VECTOR PROBLEM
# ============================================================

# This is VERY important.
#
#
# Suppose our vocabulary contains:
#
#     10,000 words
#
#
# Therefore every message must technically be represented
# using 10,000 numbers.
#
#
# Example vocabulary:
#
# [
#     "free",
#     "money",
#     "win",
#     "call",
#     "hello",
#     ...
#     10,000 words
# ]
#
#
# Our message might contain only:
#
#     "free money"
#
#
# Therefore the vector is conceptually:
#
# [
#     0.5,
#     0.4,
#     0,
#     0,
#     0,
#     0,
#     0,
#     ...
#     0
# ]
#
#
# Only a few positions contain useful values.
#
# Most positions are ZERO.
#
#
# This is called a:
#
#             SPARSE VECTOR
#
#
# "Sparse" means:
#
#     Most values are zero.
#
#
# ============================================================
#
# Why is this a problem?
#
# Suppose:
#
# vocabulary = 10,000 words
#
# messages = 5,574
#
# Then a dense matrix would contain:
#
#     5574 × 10000
#
# numbers.
#
#
# That's:
#
#     55,740,000 values
#
#
# Most of those values may be zero.
#
# So storing every zero wastes memory.
#
#
# More importantly, feeding huge vectors into our MLP
# increases the number of parameters.
#
#
# Suppose:
#
#     input = 10,000
#     first hidden layer = 16 neurons
#
# Then the first layer alone requires:
#
#     10,000 × 16
#
#     = 160,000 weights
#
#
# That's much larger than our tiny micrograd network.
# ============================================================


# ============================================================
# PART 14
# DEMONSTRATE SPARSITY
# ============================================================

# Let's create a tiny vocabulary so we can clearly see
# the zeros.
#
small_vocabulary = [
    "free",
    "money",
    "win",
    "call",
    "hello",
    "tomorrow",
    "prize",
    "claim",
    "urgent",
    "meeting"
]


def tfidf_vector(tokens, vocabulary, idf):

    # Start with a vector full of zeros.
    #
    # If vocabulary has 10 words:
    #
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #
    vector = [0.0] * len(vocabulary)


    # Calculate TF for this document.
    tf = term_frequency(tokens)


    # Go through each word in the document.
    for word, tf_value in tf.items():

        # Ignore words outside our vocabulary.
        if word not in vocabulary:
            continue


        # Find the position of this word
        # in the vocabulary.
        index = vocabulary.index(word)


        # Calculate TF-IDF.
        value = (
            tf_value *
            idf.get(word, 0)
        )


        # Store the value in the correct position.
        vector[index] = value


    return vector


# ============================================================
# CREATE A SMALL IDF FOR OUR SMALL VOCABULARY
# ============================================================

small_documents = [
    ["free", "money"],
    ["free", "prize"],
    ["call", "tomorrow"],
    ["meeting", "tomorrow"],
    ["win", "free", "money"]
]


small_idf = inverse_document_frequency(
    small_documents
)


# ============================================================
# CREATE A VECTOR
# ============================================================

test_message = [
    "free",
    "money"
]


vector = tfidf_vector(
    test_message,
    small_vocabulary,
    small_idf
)


print("\n")
print("=" * 70)
print("11. SPARSE VECTOR EXAMPLE")
print("=" * 70)


print("Vocabulary:")
print(small_vocabulary)

print("\nMessage:")
print(test_message)

print("\nTF-IDF vector:")
print(vector)


# ============================================================
# COUNT ZERO VALUES
# ============================================================

zero_count = vector.count(0.0)

non_zero_count = len(vector) - zero_count


print("\nTotal dimensions:")
print(len(vector))

print("Zero values:")
print(zero_count)

print("Non-zero values:")
print(non_zero_count)


# ============================================================
# SPARSITY RATIO
# ============================================================

# Sparsity ratio tells us what percentage
# of the vector is zero.
#
# Example:
#
# 10 dimensions
# 8 zeros
#
# sparsity = 80%
#
sparsity = (
    zero_count /
    len(vector)
) * 100


print("Sparsity:")
print(f"{sparsity:.2f}%")

"""
The most important conceptual chain is:

MESSAGE
   ↓
TOKENS
   ↓
TF
   ↓
DF
   ↓
IDF
   ↓
TF × IDF
   ↓
TF-IDF VECTOR
TF asks:

How often does this word appear in this message?

"free free money"

free → 2/3
money → 1/3
DF asks:

How many different messages contain this word?

"free"

message 1 → yes
message 2 → yes
message 3 → no

DF = 2
IDF asks:

How uncommon is this word across the whole dataset?

common word
    ↓
low IDF

rare word
    ↓
higher IDF
TF-IDF combines them:
TF-IDF = TF × IDF

So the representation considers both:

local importance
      +
global rarity
"""
