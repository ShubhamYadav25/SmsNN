"""
TRAIN DATA
   ↓
Model learns patterns
   ↓
VALIDATION DATA
   ↓
We tune decisions
   ↓
TEST DATA
   ↓
Final unseen evaluation

Training set

The model is allowed to learn from this.

Example:

80% of messages

We use it to learn:

vocabulary
IDF
neural-network weights
biases
Validation set

The model doesn't directly train its weights on this.

We use it to make decisions such as:

threshold
architecture
hyperparameters
feature choices

Think:

"Am I building the model correctly?"

Test set

This is the final exam.

We should use it only at the end.

Think:

"How well does my finished system perform on completely unseen data?"

RAW DATA
   │
   ▼
SPLIT
   │
   ├──────── TRAIN
   │
   ├──────── VALIDATION
   │
   └──────── TEST
             

TRAIN ONLY
   │
   ├── build vocabulary
   └── calculate IDF

              │
              ▼

       TRANSFORM TRAIN
       TRANSFORM VALIDATION
       TRANSFORM TEST


"""
"""
PHASE 5
-------
Train / Validation / Test Split
Data Leakage
Stratification
Reproducibility

The goal of this file is NOT to train our neural network yet.

We are learning one of the most important ML concepts:

    How do we separate data so that our evaluation is honest?

Our pipeline will eventually look like:

    RAW DATA
        ↓
    TRAIN / VALIDATION / TEST SPLIT
        ↓
    Learn vocabulary + IDF from TRAIN ONLY
        ↓
    Transform all three datasets
        ↓
    Train neural network
        ↓
    Validate / tune
        ↓
    Final TEST evaluation
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import random


# ============================================================
# 2. LOAD DATA
# ============================================================

"""
Our dataset is the UCI SMS Spam Collection.

Each row contains:

    label
    message

Example:

    ham     Hey, how are you?
    spam    Congratulations! You won a prize!

We are loading the raw dataset first.
"""

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


print("=" * 60)
print("DATASET")
print("=" * 60)

print("Total messages:", len(df))
print()


# ============================================================
# 3. LOOK AT THE ORIGINAL CLASS DISTRIBUTION
# ============================================================

"""
Before splitting, let's understand our classes.

value_counts() tells us how many examples belong to
each class.
"""

print("Original class distribution:")
print(df["label"].value_counts())

print()


"""
normalize=True gives percentages.

This is important because our dataset is imbalanced.

We have more ham messages than spam messages.
"""

print("Original class percentages:")
print(df["label"].value_counts(normalize=True) * 100)

print()


# ============================================================
# 4. WHY STRATIFICATION?
# ============================================================

"""
Imagine we have:

    5000 ham
     500 spam

Total:

    5500 messages

If we randomly split without thinking about class distribution,
we might get an unusual split.

For example:

    TRAIN:
        ham  = 90%
        spam = 10%

    TEST:
        ham  = 98%
        spam = 2%

Now test accuracy could be misleading.

Instead, we want every dataset to have approximately
the same class proportions.

This is called:

    STRATIFICATION
"""


# ============================================================
# 5. REPRODUCIBLE RANDOMNESS
# ============================================================

"""
Randomness is useful for creating our split.

But during development we want the SAME random split
every time we run the program.

Therefore we use a fixed seed.

42 is just a commonly used example.

There is nothing mathematically special about 42.
"""

random.seed(42)


# ============================================================
# 6. DEFINE SPLIT FUNCTION
# ============================================================

def stratified_split(
    dataframe,
    train_ratio=0.80,
    validation_ratio=0.10,
    test_ratio=0.10
):
    """
    Split the dataset while approximately preserving
    the class distribution.

    Example:

        80% train
        10% validation
        10% test

    The function works separately for each label.

    If we have:

        ham  -> split ham messages
        spam -> split spam messages

    Then we combine the pieces.

    IMPORTANT:

    We are ONLY splitting here.

    We are NOT:

        - building vocabulary
        - calculating IDF
        - training the model
        - looking at test answers

    This is important for avoiding leakage.
    """

    # --------------------------------------------------------
    # Check that ratios add up to 1
    # --------------------------------------------------------

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "Train, validation and test ratios must add up to 1."
        )


    # --------------------------------------------------------
    # Store the resulting pieces here
    # --------------------------------------------------------

    train_parts = []
    validation_parts = []
    test_parts = []


    # --------------------------------------------------------
    # Process each class separately
    # --------------------------------------------------------

    """
    Suppose our labels are:

        ham
        spam

    We take all ham messages and split them.

    Then we take all spam messages and split them.

    This preserves approximately the same ratio.
    """

    for label in dataframe["label"].unique():

        # Get only messages belonging to this label.
        class_data = dataframe[
            dataframe["label"] == label
        ].copy()


        # ----------------------------------------------------
        # Shuffle this class
        # ----------------------------------------------------

        """
        We shuffle because otherwise the original ordering
        of the dataset could influence our split.
        """

        indices = list(class_data.index)

        random.shuffle(indices)

        class_data = class_data.loc[indices]


        # ----------------------------------------------------
        # Calculate split positions
        # ----------------------------------------------------

        n = len(class_data)

        train_end = int(n * train_ratio)

        validation_end = train_end + int(
            n * validation_ratio
        )


        # ----------------------------------------------------
        # Slice the data
        # ----------------------------------------------------

        train_class = class_data.iloc[
            :train_end
        ]

        validation_class = class_data.iloc[
            train_end:validation_end
        ]

        test_class = class_data.iloc[
            validation_end:
        ]


        # ----------------------------------------------------
        # Store each class piece
        # ----------------------------------------------------

        train_parts.append(train_class)

        validation_parts.append(
            validation_class
        )

        test_parts.append(test_class)


    # ========================================================
    # 7. COMBINE CLASSES
    # ========================================================

    """
    At this point we have something like:

        train_parts:

            [all ham train messages]
            [all spam train messages]

    We combine them into one DataFrame.
    """

    train_df = pd.concat(
        train_parts
    )

    validation_df = pd.concat(
        validation_parts
    )

    test_df = pd.concat(
        test_parts
    )


    # ========================================================
    # 8. SHUFFLE THE FINAL DATASETS
    # ========================================================

    """
    We split by class first to preserve proportions.

    But that means the resulting dataframe could have
    class blocks.

    So we shuffle each final dataset again.

    random_state=42 makes pandas' shuffle reproducible.
    """

    train_df = train_df.sample(
        frac=1,
        random_state=42
    )

    validation_df = validation_df.sample(
        frac=1,
        random_state=42
    )

    test_df = test_df.sample(
        frac=1,
        random_state=42
    )


    # ========================================================
    # 9. RESET INDEX
    # ========================================================

    """
    After splitting, the original dataframe indices remain.

    For example:

        13
        500
        72
        1045

    Those numbers refer to the original dataframe.

    They are not useful anymore.

    reset_index(drop=True) gives each split a clean index:

        0
        1
        2
        3
        ...

    drop=True means:

        Don't keep the old index as a new column.
    """

    train_df = train_df.reset_index(drop=True)

    validation_df = validation_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)


    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# 10. PERFORM THE SPLIT
# ============================================================

train_df, validation_df, test_df = stratified_split(
    df,
    train_ratio=0.80,
    validation_ratio=0.10,
    test_ratio=0.10
)


# ============================================================
# 11. CHECK DATASET SIZES
# ============================================================

print("=" * 60)
print("SPLIT SIZES")
print("=" * 60)

print("Total:", len(df))
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))

print()


# ============================================================
# 12. CHECK CLASS DISTRIBUTION
# ============================================================

"""
This is an important debugging checkpoint.

We want approximately the same class ratio in:

    original
    train
    validation
    test
"""

print("=" * 60)
print("TRAIN CLASS DISTRIBUTION")
print("=" * 60)

print(train_df["label"].value_counts())

print()

print(
    train_df["label"].value_counts(normalize=True) * 100
)

print()


print("=" * 60)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 60)

print(validation_df["label"].value_counts())

print()

print(
    validation_df["label"].value_counts(normalize=True) * 100
)

print()


print("=" * 60)
print("TEST CLASS DISTRIBUTION")
print("=" * 60)

print(test_df["label"].value_counts())

print()

print(
    test_df["label"].value_counts(normalize=True) * 100
)

print()


# ============================================================
# 13. SHOW EXAMPLES
# ============================================================

"""
Let's make sure the actual messages are still there.

The split should NOT modify our message content.
"""

print("=" * 60)
print("TRAIN EXAMPLES")
print("=" * 60)

print(train_df.head())

print()


print("=" * 60)
print("VALIDATION EXAMPLES")
print("=" * 60)

print(validation_df.head())

print()


print("=" * 60)
print("TEST EXAMPLES")
print("=" * 60)

print(test_df.head())

print()


# ============================================================
# 14. CHECK FOR OVERLAPPING ROWS
# ============================================================

"""
A message should belong to only ONE split.

We don't want:

    same message in train
    AND validation

or:

    same message in train
    AND test

That would make evaluation less trustworthy.

We can check using the original dataframe index,
but we reset the index above.

Therefore we check message + label combinations.
"""

train_messages = set(
    zip(
        train_df["label"],
        train_df["message"]
    )
)

validation_messages = set(
    zip(
        validation_df["label"],
        validation_df["message"]
    )
)

test_messages = set(
    zip(
        test_df["label"],
        test_df["message"]
    )
)


# ------------------------------------------------------------
# Check train vs validation
# ------------------------------------------------------------

train_validation_overlap = (
    train_messages
    & validation_messages
)


# ------------------------------------------------------------
# Check train vs test
# ------------------------------------------------------------

train_test_overlap = (
    train_messages
    & test_messages
)


# ------------------------------------------------------------
# Check validation vs test
# ------------------------------------------------------------

validation_test_overlap = (
    validation_messages
    & test_messages
)


print("=" * 60)
print("OVERLAP CHECK")
print("=" * 60)

print(
    "Train / Validation overlap:",
    len(train_validation_overlap)
)

print(
    "Train / Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation / Test overlap:",
    len(validation_test_overlap)
)

print()


# ============================================================
# 15. FINAL CHECK
# ============================================================

if (
    len(train_validation_overlap) == 0
    and len(train_test_overlap) == 0
    and len(validation_test_overlap) == 0
):
    print("SUCCESS: No overlap detected between splits.")
else:
    print("WARNING: Overlap detected!")


print()


# ============================================================
# 16. VERY IMPORTANT:
#     WHAT WE MUST NOT DO YET
# ============================================================

"""
DO NOT do this before the split:

    vocabulary = build_vocabulary(df)

    idf = calculate_idf(df)

Why?

Because df contains:

    train
    validation
    test

Therefore the preprocessing would have already seen
information from validation and test.

That is DATA LEAKAGE.

The correct future pipeline is:

    1. Load raw dataset

    2. Split dataset

    3. Use TRAIN ONLY to learn:
           vocabulary
           IDF
           other preprocessing statistics

    4. Transform TRAIN

    5. Transform VALIDATION
       using the TRAIN vocabulary/IDF

    6. Transform TEST
       using the TRAIN vocabulary/IDF

    7. Train neural network using TRAIN

    8. Use VALIDATION while developing

    9. Use TEST only for final evaluation


This distinction is extremely important.

The test set is supposed to behave like completely
unseen real-world data.
"""


# ============================================================
# 17. SAVE THE SPLITS
# ============================================================

"""
Saving the splits is convenient while learning.

We can inspect them later.

However, in a larger production pipeline we may instead
recreate the split deterministically from the original data.
"""

train_df.to_csv(
    "train.csv",
    index=False
)

validation_df.to_csv(
    "validation.csv",
    index=False
)

test_df.to_csv(
    "test.csv",
    index=False
)


print("Saved:")
print("  train.csv")
print("  validation.csv")
print("  test.csv")

print()


# ============================================================
# END
# ============================================================

print("=" * 60)
print("PHASE 5 COMPLETE")
print("=" * 60)

print(
    "We have separated the data without intentionally "
    "leaking preprocessing information across splits."
)

"""
              RAW DATA
                  │
                  ▼
          ┌───────────────┐
          │     SPLIT     │
          └───────────────┘
             │     │     │
             ▼     ▼     ▼
           TRAIN  VAL   TEST
             │
             ▼
       LEARN PARAMETERS
             │
       ┌─────┴─────┐
       ▼           ▼
   Vocabulary     IDF
       │           │
       └─────┬─────┘
             ▼
       Transform all
          datasets
"""

