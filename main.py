"""
============================================================
SMS SPAM / SCAM DETECTOR
============================================================

This is the FINAL learning implementation of our project.

Dataset:
    UCI SMS Spam Collection

Goal:
    Learn to classify an SMS as:

        0 = HAM
        1 = SPAM

IMPORTANT:
    The UCI dataset is a SPAM dataset, not a pure scam dataset.

The concepts learned in this project are:

    Phase 10  - Training diagnostics
    Phase 11  - Numerical gradient checking
    Phase 12  - Evaluation metrics
    Phase 13  - Threshold experiments
    Phase 14  - Mini-batch training
    Phase 15  - Error analysis
    Phase 16  - Feature improvements
    Phase 17  - Baselines
    Phase 18  - Final detector
    Phase 19  - Visualization
    Phase 20  - Final refactor / knowledge checkpoint

NO:
    PyTorch
    TensorFlow
    Keras
    sklearn classifier
    pretrained model
    automatic differentiation

We implement the neural network ourselves.

The only external numerical help is NumPy.

============================================================
FINAL PIPELINE
============================================================

RAW SMS
   |
   v
clean duplicates
   |
   v
TRAIN / VALIDATION / TEST
   |
   +-------------------------------+
   |                               |
   v                               |
TRAIN ONLY                         |
   |                               |
   +--> vocabulary                 |
   +--> IDF                        |
                                   |
   v                               |
TF-IDF ----------------------------+
   |
   +--> handcrafted features
   |
   v
numerical X
   |
   v
OUR MLP
   |
   v
logit
   |
   v
sigmoid
   |
   v
P(spam)
   |
   v
BCE loss
   |
   v
manual backpropagation
   |
   v
mini-batch gradient descent
   |
   v
evaluation
   |
   v
final detector

============================================================
"""


# ============================================================
# IMPORTS
# ============================================================

import math
import random
import re
import string

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

# Reproducibility.
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATA_FILE = "SMSSpamCollection"


# ------------------------------------------------------------
# Split
# ------------------------------------------------------------

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ------------------------------------------------------------
# Text representation
# ------------------------------------------------------------

# We deliberately limit vocabulary size.

# Why?

# The raw vocabulary can be several thousand words.
#
# A smaller vocabulary:
#
#     - makes training faster
#     - reduces memory
#     - lets us understand the experiment
#
# Later, this can be increased.

MAX_VOCAB_SIZE = 2000


# ------------------------------------------------------------
# Neural network
# ------------------------------------------------------------

HIDDEN_1 = 32
HIDDEN_2 = 16

LEARNING_RATE = 0.01

EPOCHS = 20

BATCH_SIZE = 32


# ------------------------------------------------------------
# Threshold
# ------------------------------------------------------------

DEFAULT_THRESHOLD = 0.50


# ============================================================
# PHASE 10
# DATA LOADING
# ============================================================

def load_dataset(path):
    """
    Load the original UCI SMS Spam Collection.

    Original format:

        ham<TAB>message
        spam<TAB>message

    There is no header.
    """

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    return df


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

def clean_dataset(df):
    """
    Remove exact duplicate rows.

    Why?

    Suppose the exact same SMS appears multiple times.

    If one copy goes into training and another identical
    copy goes into testing, the model effectively sees
    the test example during training.

    That can make evaluation overly optimistic.

    Therefore we remove exact duplicates BEFORE splitting.
    """

    before = len(df)

    df = df.drop_duplicates(
        subset=["label", "message"]
    ).reset_index(drop=True)

    after = len(df)

    print("Duplicate rows removed:", before - after)

    return df


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(
    df,
    train_ratio=0.80,
    validation_ratio=0.10,
    test_ratio=0.10
):
    """
    Split while preserving the approximate ham/spam ratio.

    We split each class separately.

    This is important because spam is the minority class.
    """

    assert abs(
        train_ratio
        + validation_ratio
        + test_ratio
        - 1.0
    ) < 1e-9


    train_parts = []
    validation_parts = []
    test_parts = []


    for label in df["label"].unique():

        class_df = df[
            df["label"] == label
        ].copy()


        # Shuffle the class.
        class_df = class_df.sample(
            frac=1,
            random_state=SEED
        )


        n = len(class_df)


        train_end = int(
            n * train_ratio
        )


        validation_end = (
            train_end
            +
            int(n * validation_ratio)
        )


        train_parts.append(
            class_df.iloc[:train_end]
        )


        validation_parts.append(
            class_df.iloc[
                train_end:validation_end
            ]
        )


        test_parts.append(
            class_df.iloc[
                validation_end:
            ]
        )


    train_df = pd.concat(
        train_parts
    ).sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


    validation_df = pd.concat(
        validation_parts
    ).sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


    test_df = pd.concat(
        test_parts
    ).sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# PHASE 4 / 6
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Simple tokenizer.

    Steps:

        lowercase
        punctuation → spaces
        split into words

    Example:

        "WIN FREE MONEY!!!"

        becomes:

        ["win", "free", "money"]

    This is intentionally simple so we understand it.
    """

    text = text.lower()


    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )


    return text.split()


# ============================================================
# BUILD VOCABULARY
# ============================================================

def build_vocabulary(
    messages,
    max_vocab_size=2000
):
    """
    Build vocabulary from TRAINING DATA ONLY.

    This is extremely important.

    Validation and test messages must NOT teach us
    which words exist in our vocabulary.
    """

    word_counts = {}


    for message in messages:

        tokens = tokenize(message)


        for word in set(tokens):

            word_counts[word] = (
                word_counts.get(word, 0)
                + 1
            )


    # --------------------------------------------------------
    # Keep the most frequent words.
    # --------------------------------------------------------

    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )


    vocabulary = [
        word
        for word, count
        in sorted_words[
            :max_vocab_size
        ]
    ]


    word_to_index = {
        word: index
        for index, word
        in enumerate(vocabulary)
    }


    return (
        vocabulary,
        word_to_index
    )


# ============================================================
# DOCUMENT FREQUENCY
# ============================================================

def calculate_document_frequency(
    messages
):
    """
    DF(word) = number of documents containing the word.

    Repeated occurrences inside one document count only once.
    """

    df = {}


    for message in messages:

        tokens = set(
            tokenize(message)
        )


        for word in tokens:

            df[word] = (
                df.get(word, 0)
                + 1
            )


    return df


# ============================================================
# IDF
# ============================================================

def calculate_idf(
    messages,
    vocabulary,
    document_frequency
):
    """
    Calculate IDF using TRAINING DATA ONLY.

    Formula:

        IDF(word)
        =
        log(N / DF(word))

    We use smoothing so unseen/rare situations remain safe.
    """

    number_of_documents = len(messages)


    idf = {}


    for word in vocabulary:

        df = document_frequency.get(
            word,
            0
        )


        # Smoothing.
        idf[word] = math.log(
            (number_of_documents + 1)
            /
            (df + 1)
        ) + 1


    return idf


# ============================================================
# TF-IDF
# ============================================================

def tfidf_vector(
    message,
    word_to_index,
    idf
):
    """
    Convert one message into a TF-IDF vector.

    Unknown words are ignored.

    Example:

        vocabulary:

            ["free", "money", "win"]

        message:

            "free bitcoin"

        "free" is known.

        "bitcoin" is unknown.

        Therefore bitcoin contributes nothing.
    """

    vector = np.zeros(
        len(word_to_index),
        dtype=np.float32
    )


    tokens = tokenize(message)


    if len(tokens) == 0:

        return vector


    counts = {}


    for word in tokens:

        if word in word_to_index:

            counts[word] = (
                counts.get(word, 0)
                + 1
            )


    total_words = len(tokens)


    for word, count in counts.items():

        tf = (
            count
            /
            total_words
        )


        index = word_to_index[word]


        vector[index] = (
            tf
            *
            idf[word]
        )


    return vector


# ============================================================
# PHASE 16
# HANDCRAFTED FEATURES
# ============================================================

def handcrafted_features(message):
    """
    Add simple human-designed signals.

    IMPORTANT:

    These are NOT replacing TF-IDF.

    They are additional features.

    Features:

        1. normalized message length
        2. word count
        3. digit ratio
        4. uppercase ratio
        5. exclamation count
        6. question count
        7. URL present
        8. currency symbol present
        9. phone-like number present
        10. number of digits

    These are useful for learning feature engineering.

    They are supporting signals, NOT proof that a message
    is malicious.
    """

    length = len(message)

    words = message.split()

    word_count = len(words)


    if length == 0:

        return np.zeros(
            10,
            dtype=np.float32
        )


    digit_count = sum(
        character.isdigit()
        for character in message
    )


    uppercase_count = sum(
        character.isupper()
        for character in message
    )


    exclamation_count = (
        message.count("!")
    )


    question_count = (
        message.count("?")
    )


    contains_url = int(
        bool(
            re.search(
                r"(http://|https://|www\.)",
                message.lower()
            )
        )
    )


    contains_currency = int(
        any(
            symbol in message
            for symbol in [
                "$",
                "£",
                "€",
                "₹"
            ]
        )
    )


    contains_phone = int(
        bool(
            re.search(
                r"\b\d{7,}\b",
                message
            )
        )
    )


    features = np.array(
        [
            length / 200.0,
            word_count / 50.0,
            digit_count / max(length, 1),
            uppercase_count / max(length, 1),
            min(exclamation_count, 10) / 10.0,
            min(question_count, 10) / 10.0,
            contains_url,
            contains_currency,
            contains_phone,
            min(digit_count, 20) / 20.0
        ],
        dtype=np.float32
    )


    return features


# ============================================================
# FIT TEXT REPRESENTATION
# ============================================================

def fit_text_representation(
    train_messages
):
    """
    Learn everything that must be learned from TRAIN ONLY.

    Returns:

        vocabulary
        word_to_index
        idf
    """

    vocabulary, word_to_index = (
        build_vocabulary(
            train_messages,
            MAX_VOCAB_SIZE
        )
    )


    document_frequency = (
        calculate_document_frequency(
            train_messages
        )
    )


    idf = calculate_idf(
        train_messages,
        vocabulary,
        document_frequency
    )


    return (
        vocabulary,
        word_to_index,
        idf
    )


# ============================================================
# TRANSFORM MESSAGES
# ============================================================

def transform_messages(
    messages,
    word_to_index,
    idf
):
    """
    Convert messages into numerical features.

    TF-IDF + handcrafted features.
    """

    X = []


    for message in messages:

        tfidf = tfidf_vector(
            message,
            word_to_index,
            idf
        )


        handcrafted = (
            handcrafted_features(
                message
            )
        )


        combined = np.concatenate(
            [
                tfidf,
                handcrafted
            ]
        )


        X.append(combined)


    return np.array(
        X,
        dtype=np.float32
    )


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_labels(labels):

    return np.array(
        [
            1 if label == "spam"
            else 0
            for label in labels
        ],
        dtype=np.float32
    )


# ============================================================
# PHASE 18
# OUR OWN MLP
# ============================================================

class MLP:
    """
    Our own fully-connected neural network.

    Architecture:

        input
          ↓
        32
          ↓
        16
          ↓
         1

    Hidden activation:

        tanh

    Output:

        raw logit

    Then outside the network:

        sigmoid(logit)
    """


    def __init__(
        self,
        input_size,
        hidden1=32,
        hidden2=16
    ):

        # ----------------------------------------------------
        # Xavier-style initialization.
        # ----------------------------------------------------

        self.W1 = (
            np.random.randn(
                input_size,
                hidden1
            )
            *
            np.sqrt(
                1 / input_size
            )
        )


        self.b1 = np.zeros(
            hidden1
        )


        self.W2 = (
            np.random.randn(
                hidden1,
                hidden2
            )
            *
            np.sqrt(
                1 / hidden1
            )
        )


        self.b2 = np.zeros(
            hidden2
        )


        self.W3 = (
            np.random.randn(
                hidden2,
                1
            )
            *
            np.sqrt(
                1 / hidden2
            )
        )


        self.b3 = np.zeros(
            1
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, X):
        """
        Forward pass.

        X
        ↓
        W1
        ↓
        z1
        ↓
        tanh
        ↓
        a1
        ↓
        W2
        ↓
        z2
        ↓
        tanh
        ↓
        a2
        ↓
        W3
        ↓
        logit
        """

        self.z1 = (
            X @ self.W1
            +
            self.b1
        )


        self.a1 = np.tanh(
            self.z1
        )


        self.z2 = (
            self.a1 @ self.W2
            +
            self.b2
        )


        self.a2 = np.tanh(
            self.z2
        )


        self.logits = (
            self.a2 @ self.W3
            +
            self.b3
        ).reshape(-1)


        return self.logits


    # ========================================================
    # SIGMOID
    # ========================================================

    def sigmoid(self, logits):

        """
        Convert logit → probability.

        sigmoid(z)
        =
        1 / (1 + exp(-z))

        We clip logits to avoid overflow.
        """

        logits = np.clip(
            logits,
            -50,
            50
        )


        return (
            1
            /
            (
                1
                +
                np.exp(-logits)
            )
        )


    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_proba(self, X):

        logits = self.forward(X)

        return self.sigmoid(
            logits
        )


    # ========================================================
    # BCE
    # ========================================================

    def loss(
        self,
        y,
        probabilities
    ):
        """
        Binary Cross Entropy.

        L =
        -[
            y log(p)
            +
            (1-y) log(1-p)
        ]

        epsilon prevents log(0).
        """

        epsilon = 1e-9


        probabilities = np.clip(
            probabilities,
            epsilon,
            1 - epsilon
        )


        losses = -(
            y * np.log(probabilities)
            +
            (1 - y)
            *
            np.log(1 - probabilities)
        )


        return np.mean(
            losses
        )


    # ========================================================
    # BACKWARD
    # ========================================================

    def backward(
        self,
        X,
        y,
        probabilities
    ):
        """
        Manual backpropagation.

        This is the vectorized version of the same chain
        rule we learned with micrograd.

        A beautiful simplification:

            dL/dlogit = probability - y

        Therefore:

            dz3 = p - y

        Then we propagate backward through:

            W3
            tanh
            W2
            tanh
            W1
        """

        batch_size = len(X)


        # ----------------------------------------------------
        # Output gradient
        # ----------------------------------------------------

        dz3 = (
            probabilities - y
        ).reshape(-1, 1)


        # ----------------------------------------------------
        # Output layer gradients
        # ----------------------------------------------------

        dW3 = (
            self.a2.T
            @
            dz3
            /
            batch_size
        )


        db3 = np.mean(
            dz3,
            axis=0
        )


        # ----------------------------------------------------
        # Backprop into hidden layer 2
        # ----------------------------------------------------

        da2 = (
            dz3
            @
            self.W3.T
        )


        # tanh derivative:

        # d tanh(z) / dz
        # =
        # 1 - tanh(z)^2

        dz2 = (
            da2
            *
            (
                1
                -
                self.a2 ** 2
            )
        )


        dW2 = (
            self.a1.T
            @
            dz2
            /
            batch_size
        )


        db2 = np.mean(
            dz2,
            axis=0
        )


        # ----------------------------------------------------
        # Backprop into hidden layer 1
        # ----------------------------------------------------

        da1 = (
            dz2
            @
            self.W2.T
        )


        dz1 = (
            da1
            *
            (
                1
                -
                self.a1 ** 2
            )
        )


        dW1 = (
            X.T
            @
            dz1
            /
            batch_size
        )


        db1 = np.mean(
            dz1,
            axis=0
        )


        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3
        }


        return gradients


    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        gradients,
        learning_rate
    ):
        """
        Gradient descent:

            parameter
            =
            parameter
            -
            learning_rate * gradient
        """

        self.W1 -= (
            learning_rate
            *
            gradients["W1"]
        )


        self.b1 -= (
            learning_rate
            *
            gradients["b1"]
        )


        self.W2 -= (
            learning_rate
            *
            gradients["W2"]
        )


        self.b2 -= (
            learning_rate
            *
            gradients["b2"]
        )


        self.W3 -= (
            learning_rate
            *
            gradients["W3"]
        )


        self.b3 -= (
            learning_rate
            *
            gradients["b3"]
        )


    # ========================================================
    # PARAMETERS
    # ========================================================

    def parameters(self):

        return [
            self.W1,
            self.b1,
            self.W2,
            self.b2,
            self.W3,
            self.b3
        ]


# ============================================================
# PHASE 14
# MINI-BATCH TRAINING
# ============================================================

def train_one_epoch(
    model,
    X,
    y,
    batch_size,
    learning_rate
):
    """
    Mini-batch training.

    Instead of:

        one example
        ↓
        update

    we do:

        batch of examples
        ↓
        forward
        ↓
        average loss
        ↓
        backward
        ↓
        update
    """

    indices = np.arange(
        len(X)
    )


    np.random.shuffle(
        indices
    )


    losses = []


    for start in range(
        0,
        len(X),
        batch_size
    ):

        batch_indices = indices[
            start:
            start + batch_size
        ]


        X_batch = X[
            batch_indices
        ]


        y_batch = y[
            batch_indices
        ]


        # Forward.
        probabilities = (
            model.predict_proba(
                X_batch
            )
        )


        # Loss.
        loss = model.loss(
            y_batch,
            probabilities
        )


        # Backprop.
        gradients = model.backward(
            X_batch,
            y_batch,
            probabilities
        )


        # Update.
        model.update(
            gradients,
            learning_rate
        )


        losses.append(
            loss
        )


    return float(
        np.mean(losses)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_loss(
    model,
    X,
    y
):
    """
    Calculate loss WITHOUT changing parameters.
    """

    probabilities = (
        model.predict_proba(X)
    )


    return model.loss(
        y,
        probabilities
    )


# ============================================================
# PHASE 12
# CONFUSION MATRIX
# ============================================================

def confusion_matrix(
    y_true,
    y_pred
):
    """
    Return:

        TN FP
        FN TP
    """

    y_true = np.asarray(
        y_true
    )


    y_pred = np.asarray(
        y_pred
    )


    tp = np.sum(
        (y_true == 1)
        &
        (y_pred == 1)
    )


    tn = np.sum(
        (y_true == 0)
        &
        (y_pred == 0)
    )


    fp = np.sum(
        (y_true == 0)
        &
        (y_pred == 1)
    )


    fn = np.sum(
        (y_true == 1)
        &
        (y_pred == 0)
    )


    return (
        int(tn),
        int(fp),
        int(fn),
        int(tp)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5
):
    """
    Calculate:

        accuracy
        precision
        recall
        F1

    These are more informative than accuracy alone
    because spam is the minority class.
    """

    predictions = (
        probabilities
        >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions
        )
    )


    total = (
        tn + fp + fn + tp
    )


    accuracy = (
        (tp + tn)
        /
        max(total, 1)
    )


    precision = (
        tp
        /
        max(tp + fp, 1)
    )


    recall = (
        tp
        /
        max(tp + fn, 1)
    )


    f1 = (
        2
        *
        precision
        *
        recall
        /
        max(
            precision + recall,
            1e-12
        )
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# PHASE 13
# THRESHOLD EXPERIMENT
# ============================================================

def threshold_experiment(
    y_true,
    probabilities
):
    """
    Test multiple decision thresholds.

    IMPORTANT:

    Changing the threshold does NOT retrain the model.

    It only changes:

        probability → class
    """

    results = []


    for threshold in np.arange(
        0.10,
        1.00,
        0.05
    ):

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )


        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2
                ),
                **metrics
            }
        )


    return pd.DataFrame(
        results
    )


# ============================================================
# PHASE 11
# NUMERICAL GRADIENT CHECKING
# ============================================================

def numerical_gradient_check(
    model,
    X,
    y,
    parameter_name="W3",
    index=(0, 0),
    epsilon=1e-5
):
    """
    Compare:

        analytical gradient
        vs
        numerical gradient

    Numerical approximation:

        f(x + e) - f(x - e)
        -------------------
              2e

    This is one of the best ways to verify that our
    manually written backpropagation is correct.
    """

    # --------------------------------------------------------
    # Small batch.
    # --------------------------------------------------------

    X_small = X[:8]

    y_small = y[:8]


    probabilities = (
        model.predict_proba(
            X_small
        )
    )


    analytical = model.backward(
        X_small,
        y_small,
        probabilities
    )[parameter_name][index]


    parameter = getattr(
        model,
        parameter_name
    )


    original_value = parameter[index]


    # --------------------------------------------------------
    # f(x + epsilon)
    # --------------------------------------------------------

    parameter[index] = (
        original_value
        +
        epsilon
    )


    plus_probabilities = (
        model.predict_proba(
            X_small
        )
    )


    loss_plus = model.loss(
        y_small,
        plus_probabilities
    )


    # --------------------------------------------------------
    # f(x - epsilon)
    # --------------------------------------------------------

    parameter[index] = (
        original_value
        -
        epsilon
    )


    minus_probabilities = (
        model.predict_proba(
            X_small
        )
    )


    loss_minus = model.loss(
        y_small,
        minus_probabilities
    )


    # Restore parameter.
    parameter[index] = (
        original_value
    )


    numerical = (
        loss_plus
        -
        loss_minus
    ) / (
        2
        *
        epsilon
    )


    difference = abs(
        analytical
        -
        numerical
    )


    relative_error = (
        difference
        /
        max(
            1e-8,
            abs(analytical)
            +
            abs(numerical)
        )
    )


    return {
        "analytical": float(
            analytical
        ),
        "numerical": float(
            numerical
        ),
        "absolute_difference": float(
            difference
        ),
        "relative_error": float(
            relative_error
        )
    }


# ============================================================
# PHASE 17
# BASELINES
# ============================================================

def majority_baseline(
    y_train,
    y_test
):
    """
    Simplest possible baseline.

    Always predict the majority class.

    If ham is 87%, this baseline can already achieve
    approximately 87% accuracy.

    This demonstrates why accuracy alone can be misleading.
    """

    majority_class = int(
        np.mean(y_train)
        >=
        0.5
    )


    predictions = np.full(
        len(y_test),
        majority_class
    )


    probabilities = (
        predictions.astype(float)
    )


    return calculate_metrics(
        y_test,
        probabilities,
        0.5
    )


# ============================================================
# SIMPLE KEYWORD BASELINE
# ============================================================

def keyword_baseline(
    messages,
    y_true
):
    """
    Very simple non-ML baseline.

    If a message contains obvious words such as:

        free
        win
        winner
        prize
        claim
        cash

    predict spam.

    This is intentionally simplistic.

    We want to compare:

        hand-written rule
        vs
        learned MLP
    """

    suspicious_words = {
        "free",
        "win",
        "winner",
        "prize",
        "claim",
        "cash",
        "urgent",
        "reward"
    }


    predictions = []


    for message in messages:

        tokens = set(
            tokenize(message)
        )


        prediction = int(
            bool(
                tokens
                &
                suspicious_words
            )
        )


        predictions.append(
            prediction
        )


    predictions = np.array(
        predictions
    )


    return calculate_metrics(
        y_true,
        predictions.astype(float),
        0.5
    )


# ============================================================
# PHASE 15
# ERROR ANALYSIS
# ============================================================

def error_analysis(
    df,
    probabilities,
    threshold=0.5,
    number_of_examples=10
):
    """
    Print false positives and false negatives.

    False Positive:

        HAM predicted as SPAM.

    False Negative:

        SPAM predicted as HAM.

    These errors are extremely useful because they tell
    us where the representation/model is failing.
    """

    predictions = (
        probabilities
        >= threshold
    ).astype(int)


    actual = (
        df["label"]
        .map(
            {
                "ham": 0,
                "spam": 1
            }
        )
        .to_numpy()
    )


    false_positive_indices = np.where(
        (actual == 0)
        &
        (predictions == 1)
    )[0]


    false_negative_indices = np.where(
        (actual == 1)
        &
        (predictions == 0)
    )[0]


    print()
    print("=" * 70)
    print("FALSE POSITIVES")
    print("=" * 70)


    for index in (
        false_positive_indices[
            :number_of_examples
        ]
    ):

        print()
        print(
            "Probability:",
            round(
                float(probabilities[index]),
                4
            )
        )

        print(
            "Message:",
            df.iloc[index]["message"]
        )


    print()
    print("=" * 70)
    print("FALSE NEGATIVES")
    print("=" * 70)


    for index in (
        false_negative_indices[
            :number_of_examples
        ]
    ):

        print()
        print(
            "Probability:",
            round(
                float(probabilities[index]),
                4
            )
        )

        print(
            "Message:",
            df.iloc[index]["message"]
        )


    print()


# ============================================================
# PHASE 18
# DETECTOR
# ============================================================

class SpamDetector:
    """
    Small wrapper around:

        tokenizer
        vocabulary
        IDF
        handcrafted features
        MLP

    This separates the final user-facing detector from
    the training code.
    """

    def __init__(
        self,
        model,
        word_to_index,
        idf
    ):

        self.model = model

        self.word_to_index = (
            word_to_index
        )

        self.idf = idf


    def vectorize(self, message):

        tfidf = tfidf_vector(
            message,
            self.word_to_index,
            self.idf
        )


        handcrafted = (
            handcrafted_features(
                message
            )
        )


        return np.concatenate(
            [
                tfidf,
                handcrafted
            ]
        ).reshape(1, -1)


    def predict(
        self,
        message,
        threshold=0.5
    ):
        """
        User-facing prediction.

        Returns:

            label
            probability
            supporting signals
        """

        X = self.vectorize(
            message
        )


        probability = float(
            self.model.predict_proba(X)[0]
        )


        label = (
            "SPAM"
            if probability >= threshold
            else "HAM"
        )


        # ----------------------------------------------------
        # Supporting signals.
        # ----------------------------------------------------

        signals = []


        lower = message.lower()


        if re.search(
            r"(http://|https://|www\.)",
            lower
        ):

            signals.append(
                "contains a URL"
            )


        if any(
            symbol in message
            for symbol in [
                "$",
                "£",
                "€",
                "₹"
            ]
        ):

            signals.append(
                "contains currency symbols"
            )


        if re.search(
            r"\b\d{7,}\b",
            message
        ):

            signals.append(
                "contains a phone-like number"
            )


        suspicious_terms = [
            "free",
            "winner",
            "win",
            "prize",
            "claim",
            "urgent",
            "reward"
        ]


        found_terms = [
            term
            for term in suspicious_terms
            if term in lower
        ]


        if found_terms:

            signals.append(
                "contains promotional/risk terms: "
                +
                ", ".join(found_terms)
            )


        return {
            "label": label,
            "probability": probability,
            "signals": signals
        }


# ============================================================
# PHASE 19
# VISUALIZATION
# ============================================================

def plot_training_history(
    history
):
    """
    Plot train and validation loss.

    This lets us visually inspect:

        learning
        overfitting
        underfitting
    """

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        history["train_loss"],
        label="Train loss"
    )


    plt.plot(
        history["validation_loss"],
        label="Validation loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "BCE loss"
    )


    plt.title(
        "Training vs Validation Loss"
    )


    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# PHASE 19
# PLOT THRESHOLD CURVES
# ============================================================

def plot_threshold_results(
    threshold_results
):

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        label="Precision"
    )


    plt.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        label="Recall"
    )


    plt.plot(
        threshold_results["threshold"],
        threshold_results["f1"],
        label="F1"
    )


    plt.xlabel(
        "Threshold"
    )


    plt.ylabel(
        "Score"
    )


    plt.title(
        "Threshold Experiment"
    )


    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(" SMS SPAM DETECTOR")
    print("=" * 70)
    print()


    # ========================================================
    # LOAD
    # ========================================================

    df = load_dataset(
        DATA_FILE
    )


    print(
        "Original dataset size:",
        len(df)
    )


    print()
    print(
        df["label"].value_counts()
    )


    # ========================================================
    # CLEAN
    # ========================================================

    df = clean_dataset(
        df
    )


    print(
        "Dataset after duplicate removal:",
        len(df)
    )


    # ========================================================
    # SPLIT
    # ========================================================

    (
        train_df,
        validation_df,
        test_df
    ) = stratified_split(
        df,
        TRAIN_RATIO,
        VALIDATION_RATIO,
        TEST_RATIO
    )


    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)


    print(
        "Train:",
        len(train_df)
    )


    print(
        "Validation:",
        len(validation_df)
    )


    print(
        "Test:",
        len(test_df)
    )


    # ========================================================
    # FIT TEXT PROCESSING ON TRAIN ONLY
    # ========================================================

    print()
    print("=" * 70)
    print("FITTING TF-IDF")
    print("=" * 70)


    (
        vocabulary,
        word_to_index,
        idf
    ) = fit_text_representation(
        train_df["message"]
    )


    print(
        "Vocabulary size:",
        len(vocabulary)
    )


    # ========================================================
    # TRANSFORM
    # ========================================================

    print()
    print(
        "Transforming training data..."
    )


    X_train = transform_messages(
        train_df["message"],
        word_to_index,
        idf
    )


    print(
        "Transforming validation data..."
    )


    X_validation = transform_messages(
        validation_df["message"],
        word_to_index,
        idf
    )


    print(
        "Transforming test data..."
    )


    X_test = transform_messages(
        test_df["message"],
        word_to_index,
        idf
    )


    # ========================================================
    # LABELS
    # ========================================================

    y_train = encode_labels(
        train_df["label"]
    )


    y_validation = encode_labels(
        validation_df["label"]
    )


    y_test = encode_labels(
        test_df["label"]
    )


    print()
    print(
        "Feature shape:",
        X_train.shape
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = MLP(
        input_size=X_train.shape[1],
        hidden1=HIDDEN_1,
        hidden2=HIDDEN_2
    )


    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)


    print(
        "Architecture:",
        X_train.shape[1],
        "→",
        HIDDEN_1,
        "→",
        HIDDEN_2,
        "→ 1"
    )


    # ========================================================
    # PHASE 11
    # GRADIENT CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("GRADIENT CHECK")
    print("=" * 70)


    gradient_result = (
        numerical_gradient_check(
            model,
            X_train,
            y_train
        )
    )


    for key, value in (
        gradient_result.items()
    ):

        print(
            f"{key}: {value:.10f}"
        )


    if gradient_result[
        "relative_error"
    ] < 1e-4:

        print(
            "Gradient check: PASS"
        )

    else:

        print(
            "Gradient check: CHECK IMPLEMENTATION"
        )


    # ========================================================
    # PHASE 10 + 14
    # TRAIN
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)


    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": []
    }


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            X_train,
            y_train,
            BATCH_SIZE,
            LEARNING_RATE
        )


        validation_loss = (
            evaluate_loss(
                model,
                X_validation,
                y_validation
            )
        )


        train_probabilities = (
            model.predict_proba(
                X_train
            )
        )


        validation_probabilities = (
            model.predict_proba(
                X_validation
            )
        )


        train_metrics = (
            calculate_metrics(
                y_train,
                train_probabilities
            )
        )


        validation_metrics = (
            calculate_metrics(
                y_validation,
                validation_probabilities
            )
        )


        history["train_loss"].append(
            train_loss
        )


        history[
            "validation_loss"
        ].append(
            validation_loss
        )


        history[
            "train_accuracy"
        ].append(
            train_metrics["accuracy"]
        )


        history[
            "validation_accuracy"
        ].append(
            validation_metrics["accuracy"]
        )


        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f}"
        )


    # ========================================================
    # PHASE 12
    # VALIDATION EVALUATION
    # ========================================================

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )
    )


    validation_metrics = (
        calculate_metrics(
            y_validation,
            validation_probabilities,
            DEFAULT_THRESHOLD
        )
    )


    print()
    print("=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)


    for key, value in (
        validation_metrics.items()
    ):

        print(
            f"{key}: {value}"
        )


    # ========================================================
    # PHASE 13
    # THRESHOLD EXPERIMENT
    # ========================================================

    threshold_results = (
        threshold_experiment(
            y_validation,
            validation_probabilities
        )
    )


    print()
    print("=" * 70)
    print("THRESHOLD EXPERIMENT")
    print("=" * 70)


    print(
        threshold_results[
            [
                "threshold",
                "precision",
                "recall",
                "f1"
            ]
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Select threshold with highest validation F1.
    # --------------------------------------------------------

    best_row = (
        threshold_results
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )


    best_threshold = float(
        best_row["threshold"]
    )


    print()
    print(
        "Selected threshold:",
        best_threshold
    )


    # ========================================================
    # PHASE 12
    #  TEST
    # ========================================================

    """
    IMPORTANT:

    We do NOT tune the threshold using TEST.

    The threshold was selected using VALIDATION.

    Now TEST is used once as the final exam.
    """

    test_probabilities = (
        model.predict_proba(
            X_test
        )
    )


    test_metrics = (
        calculate_metrics(
            y_test,
            test_probabilities,
            best_threshold
        )
    )


    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)


    for key, value in (
        test_metrics.items()
    ):

        print(
            f"{key}: {value}"
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)


    print(
        "             Predicted"
    )


    print(
        "             HAM   SPAM"
    )


    print(
        f"Actual HAM  {test_metrics['tn']:4d} "
        f"{test_metrics['fp']:5d}"
    )


    print(
        f"Actual SPAM {test_metrics['fn']:4d} "
        f"{test_metrics['tp']:5d}"
    )


    # ========================================================
    # PHASE 17
    # BASELINES
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINES")
    print("=" * 70)


    majority_result = majority_baseline(
        y_train,
        y_test
    )


    print()
    print("Majority baseline:")


    print(
        majority_result
    )


    keyword_result = keyword_baseline(
        test_df["message"],
        y_test
    )


    print()
    print("Keyword baseline:")


    print(
        keyword_result
    )


    # ========================================================
    # PHASE 15
    # ERROR ANALYSIS
    # ========================================================

    error_analysis(
        test_df,
        test_probabilities,
        best_threshold,
        number_of_examples=10
    )


    # ========================================================
    # PHASE 19
    # VISUALIZATION
    # ========================================================

    plot_training_history(
        history
    )


    plot_threshold_results(
        threshold_results
    )


    # ========================================================
    # FINAL DETECTOR
    # ========================================================

    detector = SpamDetector(
        model,
        word_to_index,
        idf
    )


    print()
    print("=" * 70)
    print("FINAL DETECTOR READY")
    print("=" * 70)


    # --------------------------------------------------------
    # Test messages.
    # --------------------------------------------------------

    example_messages = [

        "Hey, are we still meeting tomorrow?",

        "Congratulations! You have won a free prize. Call now!",

        "Can you send me the project file?",

        "URGENT! Claim your FREE cash reward now!",

        "Your account needs verification. Visit https://example.com",

        "I will reach home at 8 tonight."
    ]


    for message in example_messages:

        result = detector.predict(
            message,
            best_threshold
        )


        print()
        print("-" * 70)


        print(
            "MESSAGE:",
            message
        )


        print(
            "PREDICTION:",
            result["label"]
        )


        print(
            "P(SPAM):",
            f"{result['probability']:.4f}"
        )


        if result["signals"]:

            print(
                "SUPPORTING SIGNALS:"
            )


            for signal in (
                result["signals"]
            ):

                print(
                    "  -",
                    signal
                )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PROJECT COMPLETE")
    print("=" * 70)


    print()
    print("We implemented:")

    print(
        "1. Real dataset"
    )

    print(
        "2. Leakage-safe split"
    )

    print(
        "3. Train-only vocabulary"
    )

    print(
        "4. Train-only IDF"
    )

    print(
        "5. TF-IDF"
    )

    print(
        "6. Handcrafted features"
    )

    print(
        "7. Our own MLP"
    )

    print(
        "8. Sigmoid"
    )

    print(
        "9. BCE"
    )

    print(
        "10. Manual backpropagation"
    )

    print(
        "11. Gradient checking"
    )

    print(
        "12. Mini-batch gradient descent"
    )

    print(
        "13. Validation monitoring"
    )

    print(
        "14. Precision / Recall / F1"
    )

    print(
        "15. Threshold selection"
    )

    print(
        "16. Confusion matrix"
    )

    print(
        "17. Error analysis"
    )

    print(
        "18. Baseline comparison"
    )

    print(
        "19. Real-message prediction"
    )

    print(
        "20. Visualization"
    )


    print()
    print(
        "The model is now ready for the final testing phase."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

