"""
PHASE 6
-------
Leakage-Safe TF-IDF Pipeline

We already learned:

    1. Tokenization
    2. Vocabulary
    3. Term Frequency (TF)
    4. Document Frequency (DF)
    5. Inverse Document Frequency (IDF)
    6. TF-IDF
    7. Train / validation / test split
    8. Data leakage

Now we combine them.

THE MOST IMPORTANT RULE:

    We LEARN vocabulary and IDF using TRAINING DATA ONLY.

Then we use those learned values to transform:

    train
    validation
    test

This prevents data leakage.
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import math
import string

import numpy as np
import pandas as pd


# ============================================================
# 2. LOAD THE THREE DATASETS
# ============================================================

"""
We already created these files in Phase 5:

    train.csv
    validation.csv
    test.csv

Each contains:

    label
    message
"""

train_df = pd.read_csv("train.csv")

validation_df = pd.read_csv("validation.csv")

test_df = pd.read_csv("test.csv")


print("=" * 70)
print("DATASET SIZES")
print("=" * 70)

print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))

print()


# ============================================================
# 3. TOKENIZATION
# ============================================================

def tokenize(message):
    """
    Convert a raw message into simple tokens.

    Example:

        "WIN FREE MONEY!!!"

    becomes approximately:

        ["win", "free", "money"]

    Steps:

        1. Convert to lowercase
        2. Remove punctuation
        3. Split on whitespace

    This is intentionally simple.

    Later we can improve our text preprocessing.
    """

    # --------------------------------------------------------
    # Convert everything to lowercase
    # --------------------------------------------------------

    message = message.lower()


    # --------------------------------------------------------
    # Remove punctuation
    # --------------------------------------------------------

    """
    string.punctuation contains characters such as:

        !
        ?
        .
        ,
        :
        ;
        ...

    We replace each punctuation character with a space.

    Why a space instead of ""?

    Consider:

        "hello,world"

    If we remove "," completely:

        "helloworld"

    That accidentally joins two words.

    Replacing it with a space gives:

        "hello world"
    """

    message = message.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )


    # --------------------------------------------------------
    # Split into words
    # --------------------------------------------------------

    tokens = message.split()


    return tokens


# ============================================================
# 4. TEST TOKENIZATION
# ============================================================

print("=" * 70)
print("TOKENIZATION TEST")
print("=" * 70)

example_message = (
    "Congratulations!!! You WON $1000. Call now!"
)

print("Original:")
print(example_message)

print()

print("Tokens:")
print(tokenize(example_message))

print()


# ============================================================
# 5. TOKENIZE TRAINING DATA
# ============================================================

"""
We tokenize every training message.

IMPORTANT:

At this stage we are preparing the data that will be
used to LEARN our vocabulary.

We are NOT using validation or test messages to build
the vocabulary.
"""

train_tokens = [
    tokenize(message)
    for message in train_df["message"]
]


# ============================================================
# 6. BUILD VOCABULARY FROM TRAINING DATA ONLY
# ============================================================

def build_vocabulary(documents):
    """
    Build a vocabulary from tokenized documents.

    documents:

        [
            ["free", "money"],
            ["win", "free", "prize"],
            ...
        ]

    We use a set first because each word only needs to
    appear once in the vocabulary.
    """

    vocabulary_set = set()


    for tokens in documents:

        for word in tokens:

            vocabulary_set.add(word)


    # --------------------------------------------------------
    # Sort vocabulary
    # --------------------------------------------------------

    """
    Why sort?

    A set has no meaningful ordering.

    But our vector needs a fixed position for every word.

    Example:

        vocabulary[0] = "call"
        vocabulary[1] = "free"
        vocabulary[2] = "money"

    Then "free" ALWAYS corresponds to position 1.

    Sorting gives us a deterministic ordering.
    """

    vocabulary = sorted(vocabulary_set)


    return vocabulary


vocabulary = build_vocabulary(train_tokens)


print("=" * 70)
print("VOCABULARY")
print("=" * 70)

print("Vocabulary size:", len(vocabulary))

print()

print("First 50 words:")
print(vocabulary[:50])

print()


# ============================================================
# 7. DOCUMENT FREQUENCY
# ============================================================

def document_frequency(documents):
    """
    Calculate Document Frequency (DF).

    DF(word) = number of documents containing the word.

    IMPORTANT:

    If a word appears 10 times inside ONE message,
    that message counts only ONCE.

    Example:

        Document:

        "free free free money"

    For DF:

        free  -> 1
        money -> 1

    because both occur in one document.
    """

    df = {}


    for tokens in documents:

        # ----------------------------------------------------
        # set(tokens) removes repeated words inside a document
        # ----------------------------------------------------

        unique_words = set(tokens)


        for word in unique_words:

            if word not in df:

                df[word] = 0

            df[word] += 1


    return df


train_df_frequency = document_frequency(
    train_tokens
)


# ============================================================
# 8. CALCULATE IDF
# ============================================================

def calculate_idf(documents, df_frequency):
    """
    Calculate Inverse Document Frequency.

    Formula:

        IDF(word)
        =
        log(
            total_documents
            /
            document_frequency(word)
        )

    Intuition:

        Word appears in almost every document
            -> low IDF

        Word appears in very few documents
            -> high IDF

    IMPORTANT:

    IDF is calculated using TRAINING DOCUMENTS ONLY.
    """

    total_documents = len(documents)

    idf = {}


    for word, count in df_frequency.items():

        idf[word] = math.log(
            total_documents / count
        )


    return idf


idf = calculate_idf(
    train_tokens,
    train_df_frequency
)


# ============================================================
# 9. INSPECT IDF
# ============================================================

print("=" * 70)
print("IDF EXAMPLES")
print("=" * 70)

example_words = [
    "free",
    "money",
    "call",
    "the"
]

for word in example_words:

    if word in idf:

        print(
            f"{word:10s} -> IDF = {idf[word]:.4f}"
        )

    else:

        print(
            f"{word:10s} -> not in training vocabulary"
        )

print()


# ============================================================
# 10. TERM FREQUENCY
# ============================================================

def term_frequency(tokens):
    """
    Calculate Term Frequency (TF).

    Formula:

        TF(word)
        =
        count(word) / total_words

    Example:

        ["free", "free", "money"]

    total words = 3

        TF(free)
        = 2 / 3

        TF(money)
        = 1 / 3
    """

    counts = {}


    for word in tokens:

        if word not in counts:

            counts[word] = 0

        counts[word] += 1


    total_words = len(tokens)


    # --------------------------------------------------------
    # Empty message protection
    # --------------------------------------------------------

    if total_words == 0:

        return {}


    tf = {}


    for word, count in counts.items():

        tf[word] = count / total_words


    return tf


# ============================================================
# 11. TF-IDF FOR ONE DOCUMENT
# ============================================================

def tfidf_for_document(tokens, idf):
    """
    Calculate TF-IDF values for ONE document.

    TF-IDF(word)
        =
        TF(word) × IDF(word)

    If a word isn't in the training vocabulary,
    we ignore it.

    This is important for validation/test messages.

    Example:

        Training vocabulary:

            ["free", "money"]

        New message:

            ["free", "bitcoin"]

        "free" -> known
        "bitcoin" -> unknown

    We use "free" and ignore "bitcoin".
    """

    tf = term_frequency(tokens)

    result = {}


    for word, tf_value in tf.items():

        if word in idf:

            result[word] = (
                tf_value * idf[word]
            )


    return result


# ============================================================
# 12. CONVERT TF-IDF DICTIONARY TO VECTOR
# ============================================================

def tfidf_vector(tokens, vocabulary, idf):
    """
    Convert one message into a numerical vector.

    Suppose vocabulary is:

        ["call", "free", "money", "win"]

    Message:

        "free money"

    The vector becomes approximately:

        [0, TF-IDF(free), TF-IDF(money), 0]

    Every message gets EXACTLY the same number of dimensions.

    This is critical because a neural network expects
    a fixed-size input.

    """

    # --------------------------------------------------------
    # Start with all zeros
    # --------------------------------------------------------

    vector = np.zeros(
        len(vocabulary),
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Calculate TF-IDF values
    # --------------------------------------------------------

    values = tfidf_for_document(
        tokens,
        idf
    )


    # --------------------------------------------------------
    # Put each value into the correct vocabulary position
    # --------------------------------------------------------

    for word, value in values.items():

        """
        vocabulary.index(word) tells us:

            "Where does this word live in our vector?"

        Example:

            vocabulary =
            ["call", "free", "money"]

            "free" -> index 1

        So:

            vector[1] = TF-IDF("free")
        """

        index = vocabulary.index(word)

        vector[index] = value


    return vector


# ============================================================
# 13. TRANSFORM ALL DATASETS
# ============================================================

def transform_messages(messages, vocabulary, idf):
    """
    Convert many messages into TF-IDF vectors.

    IMPORTANT:

    This function does NOT learn anything.

    It only transforms using:

        vocabulary
        IDF

    that were learned from training data.
    """

    vectors = []


    for message in messages:

        tokens = tokenize(message)

        vector = tfidf_vector(
            tokens,
            vocabulary,
            idf
        )

        vectors.append(vector)


    return np.array(vectors)


# ============================================================
# 14. TRANSFORM TRAIN
# ============================================================

"""
The training data is transformed using:

    TRAIN vocabulary
    TRAIN IDF
"""

X_train = transform_messages(
    train_df["message"],
    vocabulary,
    idf
)


# ============================================================
# 15. TRANSFORM VALIDATION
# ============================================================

"""
IMPORTANT:

We DO NOT build a new vocabulary here.

We DO NOT calculate new IDF here.

We use:

    vocabulary learned from TRAIN
    IDF learned from TRAIN
"""

X_validation = transform_messages(
    validation_df["message"],
    vocabulary,
    idf
)


# ============================================================
# 16. TRANSFORM TEST
# ============================================================

"""
Same rule:

    TEST does not teach the preprocessing anything.

It only gets transformed using the training representation.
"""

X_test = transform_messages(
    test_df["message"],
    vocabulary,
    idf
)


# ============================================================
# 17. CHECK VECTOR SHAPES
# ============================================================

print("=" * 70)
print("VECTOR SHAPES")
print("=" * 70)

print("X_train shape:", X_train.shape)

print("X_validation shape:", X_validation.shape)

print("X_test shape:", X_test.shape)

print()


# ============================================================
# 18. WHY THE SHAPE MATTERS
# ============================================================

"""
Suppose:

    vocabulary size = 7,000

Then every message becomes:

    [x1, x2, x3, ..., x7000]

Therefore:

    X_train.shape

might be:

    (4459, 7000)

Meaning:

    4459 messages
    7000 numerical features per message

Our neural network will eventually receive:

    one row = one message

    one column = one vocabulary feature
"""


# ============================================================
# 19. INSPECT ONE MESSAGE
# ============================================================

print("=" * 70)
print("ONE MESSAGE → TF-IDF VECTOR")
print("=" * 70)

message = train_df.iloc[0]["message"]

tokens = tokenize(message)

vector = X_train[0]


print("Message:")
print(message)

print()

print("Tokens:")
print(tokens)

print()

print("Vector length:")
print(len(vector))

print()

print("Number of non-zero values:")
print(np.count_nonzero(vector))

print()

print("First 30 vector values:")
print(vector[:30])

print()


# ============================================================
# 20. SPARSITY
# ============================================================

"""
We can now measure the sparse-vector problem directly.

Suppose:

    vocabulary = 7,000

and one message contains only a few words.

Most of the 7,000 values will be zero.

Sparsity tells us approximately how much of the vector
contains zeros.

Formula:

    sparsity
    =
    number_of_zero_values
    /
    total_values
"""

total_values = vector.size

zero_values = np.sum(
    vector == 0
)

sparsity = (
    zero_values
    /
    total_values
)


print("=" * 70)
print("SPARSITY")
print("=" * 70)

print("Total vector values:", total_values)

print("Zero values:", zero_values)

print(
    f"Sparsity: {sparsity * 100:.2f}%"
)

print()


# ============================================================
# 21. DENSITY
# ============================================================

"""
Density is the opposite idea.

Density:

    non-zero values
    /
    total values
"""

non_zero_values = np.count_nonzero(
    vector
)

density = (
    non_zero_values
    /
    total_values
)


print(
    f"Density: {density * 100:.2f}%"
)

print()


# ============================================================
# 22. LABELS → NUMBERS
# ============================================================

"""
Our neural network eventually needs numerical targets.

Currently:

    ham
    spam

We convert:

    ham  -> 0
    spam -> 1

Why?

Because our final neuron will eventually produce
a number between 0 and 1.

For example:

    0.02 -> likely ham
    0.91 -> likely spam

We will learn sigmoid, probability and threshold
in the next stages.
"""

label_map = {
    "ham": 0,
    "spam": 1
}


y_train = train_df["label"].map(
    label_map
).to_numpy()

y_validation = validation_df["label"].map(
    label_map
).to_numpy()

y_test = test_df["label"].map(
    label_map
).to_numpy()


# ============================================================
# 23. CHECK LABELS
# ============================================================

print("=" * 70)
print("LABELS")
print("=" * 70)

print("First 20 training labels:")

print(y_train[:20])

print()

print("Training label shape:")
print(y_train.shape)

print()


# ============================================================
# 24. FINAL DATA PIPELINE
# ============================================================

"""
At this point we have:

    X_train
        numerical TF-IDF features

    y_train
        0 / 1 labels

    X_validation
        numerical TF-IDF features

    y_validation
        0 / 1 labels

    X_test
        numerical TF-IDF features

    y_test
        0 / 1 labels


Our pipeline now looks like:

             RAW SMS
                │
                ▼
          SPLIT DATASET
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      TRAIN    VAL      TEST
        │
        ▼
   TOKENIZATION
        │
        ▼
   VOCABULARY
        │
        ▼
       IDF
        │
        ├───────────────┐
        ▼               ▼
    TRANSFORM        TRANSFORM
    TRAIN            VAL / TEST
        │               │
        └───────┬───────┘
                ▼
            TF-IDF
                │
                ▼
         NUMERICAL INPUT
                │
                ▼
             YOUR MLP
"""


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("PHASE 6 COMPLETE")
print("=" * 70)

print("Vocabulary size:", len(vocabulary))

print()

print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)
print("X_test:", X_test.shape)

print()

print("y_train:", y_train.shape)
print("y_validation:", y_validation.shape)
print("y_test:", y_test.shape)

print()

print("IMPORTANT:")
print("Vocabulary was learned from TRAIN only.")
print("IDF was calculated from TRAIN only.")
print("Validation and test were only transformed.")

print()

print("TF-IDF pipeline is ready for the neural network.")

