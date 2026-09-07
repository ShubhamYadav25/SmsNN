import pandas as pd
import string
from collections import Counter


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(message):

    # Convert text to lowercase.
    message = message.lower()

    # Remove punctuation.
    message = message.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Split into words.
    tokens = message.split()

    return tokens


# ============================================================
# TEST TOKENIZER ON REAL DATA
# ============================================================

for message in df["message"].head(5):

    print("\nOriginal:")
    print(message)

    print("Tokens:")
    print(tokenize(message))
    
    
"""
Vocabulary

Think of vocabulary as:

All unique tokens that our model knows about.
"""
# ============================================================
# BUILD VOCABULARY
# ============================================================

vocabulary = set()


# Go through every message in the dataset.
for message in df["message"]:

    # Convert the message into tokens.
    tokens = tokenize(message)

    # Add every token to the set.
    #
    # A set automatically keeps only unique values.
    #
    # If "free" appears 500 times,
    # the vocabulary still contains "free" only once.
    vocabulary.update(tokens)


print("Vocabulary size:", len(vocabulary))

print("\nFirst 50 words:")

# Sort only for easier human inspection.
print(sorted(vocabulary)[:50])

"""
Word Frequency

Vocabulary tells us:

Which words exist?

But we also want:

How often does each word occur?

Suppose:

"free" → 800 times
"money" → 500 times
"meeting" → 300 times
"hello" → 200 times

This is word frequency.
"""

# ============================================================
# WORD FREQUENCY
# ============================================================

word_counter = Counter()


# Process every message.
for message in df["message"]:

    # Convert message into tokens.
    tokens = tokenize(message)

    # Add the tokens to our Counter.
    word_counter.update(tokens)


# Show the 20 most common words.
print("\nMost common words:")

for word, count in word_counter.most_common(20):

    print(f"{word:20} {count}")
    
"""
Bag of Words

Suppose our vocabulary is:

[
    "free",
    "money",
    "win",
    "call"
]

Our message:

"free money"

contains:

free  → 1
money → 1
win   → 0
call  → 0

Therefore:

"free money"

becomes:

[1, 1, 0, 0]

This is called:

Bag of Words
"""

# ============================================================
# SIMPLE BAG OF WORDS
# ============================================================

vocabulary = [
    "free",
    "money",
    "win",
    "call"
]


def bag_of_words(message, vocabulary):

    # Convert the message into tokens.
    tokens = tokenize(message)

    # Create a vector containing one 0 for
    # every vocabulary word.
    #
    # Vocabulary has 4 words:
    #
    # [0, 0, 0, 0]
    #
    vector = [0] * len(vocabulary)

    # Check every token in the message.
    for token in tokens:

        # If this word exists in our vocabulary...
        if token in vocabulary:

            # Find where that word is located.
            index = vocabulary.index(token)

            # Increase its count.
            vector[index] += 1

    return vector


message = "free money"

vector = bag_of_words(
    message,
    vocabulary
)

print("Message:")
print(message)

print("\nVocabulary:")
print(vocabulary)

print("\nBag of Words:")
print(vector)


"""
              TEXT
                ↓
           TOKENIZATION
                ↓
             TOKENS
                ↓
           VOCABULARY
                ↓
          WORD COUNTS
                ↓
         BAG OF WORDS
                ↓
       NUMERICAL VECTOR
                ↓
        ┌──────────────┐
        │ NEURAL       │
        │ NETWORK      │
        └──────────────┘
"""
