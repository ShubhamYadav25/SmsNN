"""
PHASE 8
-------
Binary Cross-Entropy Loss

We are learning how to measure whether a binary
classification prediction is good or bad.

Our project:

    0 = HAM
    1 = SPAM

The model produces:

    p = probability of SPAM

BCE tells us how good that probability was.

Main formula:

    L = -[ y*log(p) + (1-y)*log(1-p) ]

Then later:

    loss
      ↓
    backpropagation
      ↓
    gradients
      ↓
    gradient descent
      ↓
    better weights
"""


# ============================================================
# 1. IMPORT
# ============================================================

import math


# ============================================================
# 2. BCE FUNCTION
# ============================================================

def binary_cross_entropy(y, p):
    """
    Calculate Binary Cross-Entropy for ONE example.

    Parameters
    ----------
    y:
        Actual label.

        0 = HAM
        1 = SPAM

    p:
        Predicted probability of SPAM.

        Must be between 0 and 1.

    Formula:

        L = -[
                y * log(p)
                +
                (1-y) * log(1-p)
            ]

    """

    # --------------------------------------------------------
    # Protect against log(0)
    # --------------------------------------------------------

    """
    Mathematically:

        log(0) = -infinity

    That would cause a numerical problem.

    Therefore we keep p slightly away from exactly
    0 and 1.

    Example:

        p = 0
        becomes approximately:
        0.000000001

        p = 1
        becomes approximately:
        0.999999999
    """

    epsilon = 1e-9

    p = max(
        epsilon,
        min(1 - epsilon, p)
    )


    # --------------------------------------------------------
    # BCE formula
    # --------------------------------------------------------

    loss = -(
        y * math.log(p)
        +
        (1 - y) * math.log(1 - p)
    )


    return loss


# ============================================================
# 3. BASIC EXAMPLES
# ============================================================

print("=" * 70)
print("BCE BASIC EXAMPLES")
print("=" * 70)


# Actual answer = SPAM
y = 1


predictions = [
    0.99,
    0.90,
    0.70,
    0.50,
    0.30,
    0.10,
    0.01
]


for p in predictions:

    loss = binary_cross_entropy(
        y,
        p
    )

    print(
        f"actual={y} "
        f"prediction={p:.2f} "
        f"loss={loss:.4f}"
    )


print()


# ============================================================
# 4. HAM EXAMPLES
# ============================================================

"""
Now the true answer is HAM.

Therefore:

    y = 0

Remember:

    p = probability of SPAM

So a small p is good.

For example:

    p = 0.01

means:

    1% probability of spam

which is good when the actual message is ham.
"""

print("=" * 70)
print("BCE WHEN ACTUAL LABEL IS HAM")
print("=" * 70)


y = 0


for p in predictions:

    loss = binary_cross_entropy(
        y,
        p
    )

    print(
        f"actual={y} "
        f"prediction={p:.2f} "
        f"loss={loss:.4f}"
    )


print()


# ============================================================
# 5. CONFIDENTLY CORRECT VS CONFIDENTLY WRONG
# ============================================================

"""
This is one of the most important properties of BCE.

Suppose:

    actual = SPAM

Then:

    p = 0.99

is confidently correct.

But:

    p = 0.01

is confidently wrong.

BCE heavily punishes the second case.
"""

print("=" * 70)
print("CONFIDENT PREDICTIONS")
print("=" * 70)


y = 1


good_prediction = 0.99

bad_prediction = 0.01


good_loss = binary_cross_entropy(
    y,
    good_prediction
)

bad_loss = binary_cross_entropy(
    y,
    bad_prediction
)


print(
    "Good prediction:",
    good_prediction
)

print(
    "Good loss:",
    good_loss
)

print()

print(
    "Bad prediction:",
    bad_prediction
)

print(
    "Bad loss:",
    bad_loss
)

print()


# ============================================================
# 6. THE SPECIAL CASE p = 0.5
# ============================================================

"""
If the model predicts:

    p = 0.5

it is completely uncertain.

For either class:

    BCE = -log(0.5)

which is approximately:

    0.6931
"""

loss_ham = binary_cross_entropy(
    0,
    0.5
)

loss_spam = binary_cross_entropy(
    1,
    0.5
)


print("=" * 70)
print("UNCERTAIN PREDICTION")
print("=" * 70)

print(
    "HAM with p=0.5:",
    loss_ham
)

print(
    "SPAM with p=0.5:",
    loss_spam
)

print()


# ============================================================
# 7. SMALL LOSS VS LARGE LOSS
# ============================================================

"""
The goal of training is:

    minimize loss

Not:

    maximize probability directly.

The probability improves because minimizing BCE
encourages the model to put probability on the correct class.
"""

examples = [
    # (actual, probability)
    (1, 0.99),
    (1, 0.80),
    (1, 0.20),
    (1, 0.01),

    (0, 0.01),
    (0, 0.20),
    (0, 0.80),
    (0, 0.99),
]


print("=" * 70)
print("LOSS COMPARISON")
print("=" * 70)


for y, p in examples:

    loss = binary_cross_entropy(
        y,
        p
    )

    if y == 1:

        actual_name = "SPAM"

    else:

        actual_name = "HAM"


    print(
        f"actual={actual_name:4s} "
        f"p(spam)={p:.2f} "
        f"loss={loss:.4f}"
    )


print()


# ============================================================
# 8. AVERAGE LOSS
# ============================================================

"""
During neural-network training we normally have
many examples.

Example:

    message 1 → loss = 0.2
    message 2 → loss = 0.7
    message 3 → loss = 0.1

We want one number representing the batch.

So we calculate the mean:

    average_loss
        =
    sum(losses) / number_of_examples
"""


batch = [
    (1, 0.9),
    (1, 0.8),
    (0, 0.1),
    (0, 0.2),
]


losses = []


for y, p in batch:

    loss = binary_cross_entropy(
        y,
        p
    )

    losses.append(loss)


average_loss = (
    sum(losses)
    /
    len(losses)
)


print("=" * 70)
print("AVERAGE BATCH LOSS")
print("=" * 70)

print("Individual losses:")

for loss in losses:

    print(
        f"{loss:.4f}"
    )

print()

print(
    f"Average loss: {average_loss:.4f}"
)

print()


# ============================================================
# 9. THE COMPLETE LEARNING IDEA
# ============================================================

"""
Our neural network produces:

    p = predicted probability

BCE compares:

    p

against:

    y = actual answer

Then:

    loss

tells us how bad the prediction was.

Training eventually does:

    loss
      ↓
    backward()
      ↓
    gradients
      ↓
    update weights
      ↓
    forward again
      ↓
    hopefully lower loss


This is the same backpropagation concept you already
learned with micrograd.

The difference is now our data is real:

    SMS messages
        ↓
    TF-IDF vectors
        ↓
    neural network
        ↓
    spam probability
        ↓
    BCE
"""


# ============================================================
# 10. IMPORTANT DIFFERENCE:
#     LOSS VS THRESHOLD
# ============================================================

"""
Do NOT confuse these two.

LOSS:

    Used for learning.

    Example:

        prediction = 0.90
        actual = 1

        BCE = small

    The optimizer tries to reduce this loss.

THRESHOLD:

    Used for making a final class decision.

    Example:

        prediction = 0.90
        threshold = 0.50

        → SPAM


The threshold does NOT train the network.

The loss DOES guide training.
"""


print("=" * 70)
print("PHASE 8 COMPLETE")
print("=" * 70)

print("BCE measures prediction error.")

print()

print("Next conceptual step:")
print("connect BCE with sigmoid and backpropagation.")

"""
              SMS
               ↓
             TF-IDF
               ↓
              x
               ↓
             MLP
               ↓
              z
               ↓
           sigmoid
               ↓
              p
               ↓
             BCE
               ↓
             loss
               ↓
          backward()
               ↓
           gradients
               ↓
        update parameters
               ↓
           better MLP
"""
