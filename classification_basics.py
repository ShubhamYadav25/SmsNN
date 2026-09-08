"""
PHASE 7
-------
From numerical features to a binary prediction.

We will learn:

    TF-IDF vector
        ↓
    weighted sum
        ↓
    logit
        ↓
    sigmoid
        ↓
    probability
        ↓
    threshold
        ↓
    HAM / SPAM

This file intentionally does NOT train a neural network yet.

We are isolating the mathematical idea first.
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import math


# ============================================================
# 2. SIGMOID
# ============================================================

def sigmoid(x):
    """
    Convert a real-valued number into a value between 0 and 1.

    Formula:

                    1
        sigmoid = -------
                  1 + e^(-x)

    Examples:

        x = -5  -> approximately 0.0067
        x =  0  -> 0.5
        x =  5  -> approximately 0.9933

    In our classifier:

        sigmoid(logit)
            =
        estimated probability of SPAM
    """

    return 1 / (1 + math.exp(-x))


# ============================================================
# 3. UNDERSTAND SIGMOID
# ============================================================

print("=" * 70)
print("SIGMOID")
print("=" * 70)

values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]

for value in values:

    probability = sigmoid(value)

    print(
        f"logit = {value:>3} "
        f"→ sigmoid = {probability:.6f}"
    )

print()


# ============================================================
# 4. A SIMPLE LINEAR MODEL
# ============================================================

"""
Before using a complete MLP, let's understand ONE neuron.

Suppose we have three features:

    x1
    x2
    x3

and three weights:

    w1
    w2
    w3

The neuron calculates:

    z = x1*w1 + x2*w2 + x3*w3 + b

This is exactly the same weighted-sum operation
you learned in micrograd.
"""

x = [
    0.2,
    0.7,
    0.1
]

w = [
    1.5,
    2.0,
    -1.0
]

b = -0.5


# ============================================================
# 5. CALCULATE THE LOGIT
# ============================================================

"""
Let's calculate:

    z = x1*w1 + x2*w2 + x3*w3 + b
"""

z = (
    x[0] * w[0]
    + x[1] * w[1]
    + x[2] * w[2]
    + b
)


print("=" * 70)
print("ONE NEURON")
print("=" * 70)

print("Inputs:", x)

print("Weights:", w)

print("Bias:", b)

print()

print("Logit:", z)

print()


# ============================================================
# 6. CONVERT LOGIT TO PROBABILITY
# ============================================================

"""
The logit is not a probability.

Now apply sigmoid.
"""

probability = sigmoid(z)


print("=" * 70)
print("LOGIT → PROBABILITY")
print("=" * 70)

print("Logit:", z)

print(
    "P(spam):",
    probability
)

print()


# ============================================================
# 7. THRESHOLD
# ============================================================

"""
Now convert probability into a class.

Our convention:

    0 = ham
    1 = spam

With threshold 0.5:

    probability >= 0.5
        → spam

    probability < 0.5
        → ham
"""

threshold = 0.5


if probability >= threshold:

    prediction = 1
    prediction_name = "SPAM"

else:

    prediction = 0
    prediction_name = "HAM"


print("=" * 70)
print("FINAL PREDICTION")
print("=" * 70)

print("Probability:", probability)

print("Threshold:", threshold)

print("Prediction:", prediction)

print("Class:", prediction_name)

print()


# ============================================================
# 8. TEST DIFFERENT LOGITS
# ============================================================

"""
This makes the relationship easier to see.

Negative logits produce probabilities below 0.5.

Positive logits produce probabilities above 0.5.

Exactly zero gives:

    sigmoid(0) = 0.5
"""

print("=" * 70)
print("LOGIT → PROBABILITY → CLASS")
print("=" * 70)

for logit in [-5, -2, -1, 0, 1, 2, 5]:

    probability = sigmoid(logit)

    if probability >= 0.5:

        class_name = "SPAM"

    else:

        class_name = "HAM"

    print(
        f"logit={logit:>3} "
        f"probability={probability:.4f} "
        f"class={class_name}"
    )

print()


# ============================================================
# 9. THRESHOLD EXPERIMENT
# ============================================================

"""
IMPORTANT:

0.5 is only one possible threshold.

Suppose the model outputs:

    probability = 0.70

With:

    threshold = 0.50

we predict:

    SPAM

But with:

    threshold = 0.80

we predict:

    HAM

So changing the threshold changes model behavior
WITHOUT changing the neural-network weights.

This becomes very important when we discuss:

    false positives
    false negatives
    precision
    recall
"""

test_probability = 0.70


print("=" * 70)
print("THRESHOLD EXPERIMENT")
print("=" * 70)

for threshold in [0.3, 0.5, 0.7, 0.9]:

    if test_probability >= threshold:

        class_name = "SPAM"

    else:

        class_name = "HAM"

    print(
        f"probability={test_probability:.2f} "
        f"threshold={threshold:.2f} "
        f"→ {class_name}"
    )

print()


# ============================================================
# 10. IMPORTANT: THRESHOLD DOES NOT TRAIN THE MODEL
# ============================================================

"""
This distinction is VERY important.

Training changes:

    weights
    biases

Thresholding does NOT change:

    weights
    biases

Thresholding only changes:

    how we convert the final probability
    into a class decision.

For example:

    Model output = 0.65

Threshold 0.50:
    → SPAM

Threshold 0.70:
    → HAM

The model itself produced exactly the same 0.65.

Only our decision rule changed.
"""


# ============================================================
# 11. NUMERICAL STABILITY NOTE
# ============================================================

"""
Our simple sigmoid:

    1 / (1 + exp(-x))

is perfect for learning.

However, for extremely large positive/negative values,
direct exponentiation can have numerical problems.

For example:

    exp(1000)

is enormous.

Later, when implementing the real training system,
we can make the sigmoid numerically safer.

For now, keep the simple equation visible because
we are learning the mathematics.
"""


# ============================================================
# 12. FINAL DATA FLOW
# ============================================================

"""
Our complete conceptual flow is now:

    MESSAGE
       ↓
    TOKENIZATION
       ↓
    TF-IDF
       ↓
    x1, x2, x3, ..., xn
       ↓
    WEIGHTED SUM
       ↓
    z
       ↓
    SIGMOID
       ↓
    P(SPAM)
       ↓
    THRESHOLD
       ↓
    HAM / SPAM


The neural network's job is mainly to learn the
weights and biases that make the probability useful.
"""


print("=" * 70)
print("PHASE 7 COMPLETE")
print("=" * 70)

print("Text has now been connected conceptually")
print("to binary classification.")
