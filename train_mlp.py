"""
train_mlp.py

PHASE 9
-------
Connect:

    TF-IDF
       ↓
    Your own MLP
       ↓
    Logit
       ↓
    Sigmoid
       ↓
    Probability
       ↓
    BCE
       ↓
    Backpropagation
       ↓
    Gradient Descent


IMPORTANT:

This file intentionally uses our own neural-network
implementation.

We are NOT using:

    PyTorch
    TensorFlow
    Keras
    sklearn neural networks
    automatic differentiation

The purpose is to understand what is happening.
"""


# ============================================================
# IMPORTS
# ============================================================

import math
import random
import string

import numpy as np
import pandas as pd


# ============================================================
# 1. MICROGRAD VALUE
# ============================================================

class Value:
    """
    A scalar value that stores:

        data
        gradient
        computational graph

    This is the core idea from micrograd.

    Every Value remembers how it was created so that
    backward() can later propagate gradients backward.
    """

    def __init__(
        self,
        data,
        _children=(),
        _op="",
        label=""
    ):

        self.data = float(data)

        # Gradient starts at zero.
        self.grad = 0.0

        # Children are the Values used to create this Value.
        self._prev = set(_children)

        # Operation that created this Value.
        self._op = _op

        # Optional human-readable label.
        self.label = label

        # Function that computes local gradients.
        self._backward = lambda: None


    # ========================================================
    # ADDITION
    # ========================================================

    def __add__(self, other):

        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )


        def _backward():

            self.grad += out.grad

            other.grad += out.grad


        out._backward = _backward

        return out


    # Support:

        # number + Value

    __radd__ = __add__


    # ========================================================
    # MULTIPLICATION
    # ========================================================

    def __mul__(self, other):

        other = (
            other
            if isinstance(other, Value)
            else Value(other)
        )

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )


        def _backward():

            self.grad += (
                other.data
                * out.grad
            )

            other.grad += (
                self.data
                * out.grad
            )


        out._backward = _backward

        return out


    # Support:

        # number * Value

    __rmul__ = __mul__


    # ========================================================
    # NEGATION
    # ========================================================

    def __neg__(self):

        return self * -1


    # ========================================================
    # SUBTRACTION
    # ========================================================

    def __sub__(self, other):

        return self + (-other)


    # Support:

        # number - Value

    def __rsub__(self, other):

        return other + (-self)


    # ========================================================
    # POWER
    # ========================================================

    def __pow__(self, exponent):

        out = Value(
            self.data ** exponent,
            (self,),
            f"**{exponent}"
        )


        def _backward():

            self.grad += (
                exponent
                * self.data ** (exponent - 1)
                * out.grad
            )


        out._backward = _backward

        return out


    # ========================================================
    # DIVISION
    # ========================================================

    def __truediv__(self, other):

        return self * (
            other ** -1
            if isinstance(other, Value)
            else Value(other) ** -1
        )


    # ========================================================
    # EXPONENTIAL
    # ========================================================

    def exp(self):

        x = self.data

        out = Value(
            math.exp(x),
            (self,),
            "exp"
        )


        def _backward():

            self.grad += (
                out.data
                * out.grad
            )


        out._backward = _backward

        return out


    # ========================================================
    # LOGARITHM
    # ========================================================

    def log(self):

        out = Value(
            math.log(self.data),
            (self,),
            "log"
        )


        def _backward():

            self.grad += (
                (1 / self.data)
                * out.grad
            )


        out._backward = _backward

        return out


    # ========================================================
    # TANH
    # ========================================================

    def tanh(self):

        t = math.tanh(
            self.data
        )

        out = Value(
            t,
            (self,),
            "tanh"
        )


        def _backward():

            self.grad += (
                (1 - t ** 2)
                * out.grad
            )


        out._backward = _backward

        return out


    # ========================================================
    # BACKWARD
    # ========================================================

    def backward(self):

        """
        Build a topological ordering of the computational graph.

        Then traverse it backward.

        This is the same backpropagation mechanism
        we learned in micrograd.
        """

        topo = []

        visited = set()


        def build_topological_graph(v):

            if v not in visited:

                visited.add(v)

                for child in v._prev:

                    build_topological_graph(child)

                topo.append(v)


        build_topological_graph(self)


        # ----------------------------------------------------
        # Start with d(loss)/d(loss) = 1
        # ----------------------------------------------------

        self.grad = 1.0


        # ----------------------------------------------------
        # Traverse graph backward
        # ----------------------------------------------------

        for node in reversed(topo):

            node._backward()


# ============================================================
# 2. SIGMOID
# ============================================================

def sigmoid(value):
    """
    Sigmoid for a Value object.

        sigmoid(z)
        =
        1 / (1 + e^(-z))

    We implement it using Value operations so that
    backpropagation can travel through sigmoid.

    This is important.

    We DON'T want to calculate sigmoid using only
    normal Python numbers because then the computational
    graph would be broken.
    """

    return (
        1
        /
        (
            1
            + (-value).exp()
        )
    )


# ============================================================
# 3. BINARY CROSS-ENTROPY
# ============================================================

def binary_cross_entropy(
    probability,
    target
):
    """
    BCE for one example.

        L =
        -[
            y*log(p)
            +
            (1-y)*log(1-p)
         ]

    target:

        0 = HAM
        1 = SPAM

    probability:

        predicted probability of SPAM
    """

    # --------------------------------------------------------
    # Tiny epsilon for numerical safety
    # --------------------------------------------------------

    epsilon = 1e-9


    # --------------------------------------------------------
    # We use Value operations here.
    # --------------------------------------------------------

    p = probability


    # --------------------------------------------------------
    # BCE formula
    # --------------------------------------------------------

    loss = -(
        target * (p + epsilon).log()
        +
        (1 - target)
        * (1 - p + epsilon).log()
    )


    return loss


# ============================================================
# 4. NEURON
# ============================================================

class Neuron:
    """
    A single neuron.

    It calculates:

        weighted_sum =
            x1*w1
            + x2*w2
            + ...
            + xn*wn
            + b

    Then optionally applies tanh.

    For the final output neuron we will NOT use tanh.

    Why?

    Because we want the output neuron to produce a
    raw LOGIT.

    Then we apply:

        sigmoid(logit)
    """

    def __init__(
        self,
        number_of_inputs,
        activation=True
    ):

        self.weights = [
            Value(
                random.uniform(-0.1, 0.1)
            )
            for _ in range(number_of_inputs)
        ]

        self.bias = Value(0.0)

        self.activation = activation


    def __call__(self, x):

        # ----------------------------------------------------
        # Weighted sum
        # ----------------------------------------------------

        result = sum(
            (
                weight * input_value
                for weight, input_value
                in zip(self.weights, x)
            ),
            self.bias
        )


        # ----------------------------------------------------
        # Hidden layers use tanh.
        # Output layer returns raw logit.
        # ----------------------------------------------------

        if self.activation:

            return result.tanh()

        return result


    def parameters(self):

        return self.weights + [
            self.bias
        ]


# ============================================================
# 5. LAYER
# ============================================================

class Layer:
    """
    A collection of neurons.

    Example:

        input size = 100
        neurons = 32

    gives:

        32 neurons

    Each neuron receives all 100 input features.
    """

    def __init__(
        self,
        number_of_inputs,
        number_of_neurons,
        activation=True
    ):

        self.neurons = [
            Neuron(
                number_of_inputs,
                activation
            )
            for _ in range(number_of_neurons)
        ]


    def __call__(self, x):

        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]

        return outputs


    def parameters(self):

        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]


# ============================================================
# 6. MLP
# ============================================================

class MLP:
    """
    Multi-Layer Perceptron.

    Example architecture:

        input
          ↓
        32
          ↓
        16
          ↓
        1

    The final layer has:

        activation=False

    because we want a raw logit.
    """

    def __init__(
        self,
        number_of_inputs,
        layer_sizes
    ):

        self.layers = []

        input_size = number_of_inputs


        for layer_index, output_size in enumerate(
            layer_sizes
        ):

            # ------------------------------------------------
            # Final layer produces LOGIT.
            # ------------------------------------------------

            is_final_layer = (
                layer_index
                ==
                len(layer_sizes) - 1
            )


            activation = not is_final_layer


            layer = Layer(
                input_size,
                output_size,
                activation=activation
            )


            self.layers.append(layer)


            input_size = output_size


    def __call__(self, x):

        for layer in self.layers:

            x = layer(x)


        # ----------------------------------------------------
        # Our final layer has one neuron.
        # ----------------------------------------------------

        return x[0]


    def parameters(self):

        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]


# ============================================================
# 7. TEXT PREPROCESSING
# ============================================================

def tokenize(message):

    """
    Same tokenizer used in the previous phase.

    We must keep preprocessing consistent.

    """

    message = message.lower()


    message = message.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )


    return message.split()


# ============================================================
# 8. BUILD VOCABULARY
# ============================================================

def build_vocabulary(documents):

    """
    Build vocabulary from TRAINING DOCUMENTS ONLY.
    """

    vocabulary_set = set()


    for tokens in documents:

        vocabulary_set.update(tokens)


    return sorted(vocabulary_set)


# ============================================================
# 9. DOCUMENT FREQUENCY
# ============================================================

def document_frequency(documents):

    df = {}


    for tokens in documents:

        for word in set(tokens):

            if word not in df:

                df[word] = 0

            df[word] += 1


    return df


# ============================================================
# 10. IDF
# ============================================================

def calculate_idf(
    documents,
    df_frequency
):

    total_documents = len(documents)

    idf = {}


    for word, count in df_frequency.items():

        idf[word] = math.log(
            total_documents / count
        )


    return idf


# ============================================================
# 11. TF
# ============================================================

def term_frequency(tokens):

    counts = {}


    for word in tokens:

        counts[word] = (
            counts.get(word, 0)
            + 1
        )


    total_words = len(tokens)


    if total_words == 0:

        return {}


    return {
        word: count / total_words
        for word, count in counts.items()
    }


# ============================================================
# 12. TF-IDF VECTOR
# ============================================================

def tfidf_vector(
    tokens,
    word_to_index,
    idf
):

    """
    Convert a tokenized message into a fixed-size
    numerical vector.

    word_to_index gives O(1)-style dictionary lookup
    instead of repeatedly calling vocabulary.index().
    """

    vector = np.zeros(
        len(word_to_index),
        dtype=np.float64
    )


    tf = term_frequency(tokens)


    for word, tf_value in tf.items():

        # ----------------------------------------------------
        # Unknown words are ignored.
        # ----------------------------------------------------

        if word not in word_to_index:

            continue


        index = word_to_index[word]


        # ----------------------------------------------------
        # TF-IDF = TF × IDF
        # ----------------------------------------------------

        vector[index] = (
            tf_value
            * idf[word]
        )


    return vector


# ============================================================
# 13. BUILD ALL TRAINING REPRESENTATION
# ============================================================

def prepare_tfidf(train_messages):

    """
    IMPORTANT:

    This function LEARNS the vocabulary and IDF.

    Therefore it must only receive TRAINING messages.

    It returns:

        vocabulary
        word_to_index
        idf
    """

    train_tokens = [
        tokenize(message)
        for message in train_messages
    ]


    vocabulary = build_vocabulary(
        train_tokens
    )


    word_to_index = {
        word: index
        for index, word
        in enumerate(vocabulary)
    }


    df_frequency = document_frequency(
        train_tokens
    )


    idf = calculate_idf(
        train_tokens,
        df_frequency
    )


    return (
        vocabulary,
        word_to_index,
        idf
    )


# ============================================================
# 14. TRANSFORM DATASET
# ============================================================

def transform_dataset(
    messages,
    word_to_index,
    idf
):

    vectors = []


    for message in messages:

        tokens = tokenize(message)


        vector = tfidf_vector(
            tokens,
            word_to_index,
            idf
        )


        vectors.append(vector)


    return np.array(vectors)


# ============================================================
# 15. LOAD DATA
# ============================================================

train_df = pd.read_csv(
    "train.csv"
)

validation_df = pd.read_csv(
    "validation.csv"
)

test_df = pd.read_csv(
    "test.csv"
)


# ============================================================
# 16. FIT TF-IDF ON TRAIN ONLY
# ============================================================

print("=" * 70)
print("FITTING TF-IDF ON TRAINING DATA")
print("=" * 70)


(
    vocabulary,
    word_to_index,
    idf
) = prepare_tfidf(
    train_df["message"]
)


print(
    "Vocabulary size:",
    len(vocabulary)
)

print()


# ============================================================
# 17. TRANSFORM ALL THREE DATASETS
# ============================================================

print("=" * 70)
print("TRANSFORMING DATA")
print("=" * 70)


X_train = transform_dataset(
    train_df["message"],
    word_to_index,
    idf
)


X_validation = transform_dataset(
    validation_df["message"],
    word_to_index,
    idf
)


X_test = transform_dataset(
    test_df["message"],
    word_to_index,
    idf
)


print("X_train:", X_train.shape)

print(
    "X_validation:",
    X_validation.shape
)

print("X_test:", X_test.shape)

print()


# ============================================================
# 18. LABELS
# ============================================================

label_map = {
    "ham": 0,
    "spam": 1
}


y_train = train_df["label"].map(
    label_map
).to_numpy()


y_validation = (
    validation_df["label"]
    .map(label_map)
    .to_numpy()
)


y_test = (
    test_df["label"]
    .map(label_map)
    .to_numpy()
)


# ============================================================
# 19. CREATE MLP
# ============================================================

"""
If vocabulary contains:

    7000 words

then:

    input size = 7000

Our architecture:

    7000 → 32 → 16 → 1
"""

input_size = X_train.shape[1]


model = MLP(
    number_of_inputs=input_size,
    layer_sizes=[
        32,
        16,
        1
    ]
)


print("=" * 70)
print("MODEL")
print("=" * 70)

print(
    "Input features:",
    input_size
)

print(
    "Architecture:",
    input_size,
    "→ 32 → 16 → 1"
)

print(
    "Number of parameters:",
    len(model.parameters())
)

print()


# ============================================================
# 20. SINGLE EXAMPLE FORWARD PASS
# ============================================================

"""
Before training the whole dataset,
let's understand ONE example.

NumPy gives us:

    X_train[0]

We convert each number to Value.

Then our MLP can construct a computational graph.
"""

def numpy_to_values(vector):

    return [
        Value(number)
        for number in vector
    ]


x_example = numpy_to_values(
    X_train[0]
)


logit = model(
    x_example
)


probability = sigmoid(
    logit
)


print("=" * 70)
print("SINGLE FORWARD PASS")
print("=" * 70)

print(
    "Message:",
    train_df.iloc[0]["message"]
)

print(
    "Actual label:",
    y_train[0]
)

print(
    "Logit:",
    logit.data
)

print(
    "Probability:",
    probability.data
)

print()


# ============================================================
# 21. SINGLE EXAMPLE LOSS
# ============================================================

loss = binary_cross_entropy(
    probability,
    y_train[0]
)


print("=" * 70)
print("SINGLE EXAMPLE LOSS")
print("=" * 70)

print(
    "Loss:",
    loss.data
)

print()


# ============================================================
# 22. BACKPROPAGATION
# ============================================================

"""
Now the important part.

loss.backward()

will traverse:

    loss
      ↓
    BCE
      ↓
    sigmoid
      ↓
    logit
      ↓
    MLP
      ↓
    weights and biases

and calculate gradients.

This is exactly the computational-graph idea
from micrograd.
"""

loss.backward()


print("=" * 70)
print("BACKPROPAGATION")
print("=" * 70)

print(
    "First 10 parameter gradients:"
)

for parameter in model.parameters()[:10]:

    print(
        f"data={parameter.data:.6f} "
        f"grad={parameter.grad:.6f}"
    )

print()


# ============================================================
# 23. RESET GRADIENTS
# ============================================================

"""
Before doing another forward/backward pass,
we must reset gradients.

Remember:

    gradients ACCUMULATE.

So:

    parameter.grad = 0

must happen before another training step.
"""

def zero_grad():

    for parameter in model.parameters():

        parameter.grad = 0.0


zero_grad()


# ============================================================
# 24. PREDICTION FUNCTION
# ============================================================

def predict_probability(
    model,
    vector
):

    """
    Run one example through the model.

        vector
          ↓
        Value objects
          ↓
        MLP
          ↓
        logit
          ↓
        sigmoid
          ↓
        probability
    """

    x = numpy_to_values(
        vector
    )


    logit = model(x)


    probability = sigmoid(
        logit
    )


    return probability.data


# ============================================================
# 25. CLASSIFICATION FUNCTION
# ============================================================

def predict_class(
    probability,
    threshold=0.5
):

    if probability >= threshold:

        return 1

    return 0


# ============================================================
# 26. CHECK MULTIPLE PREDICTIONS
# ============================================================

print("=" * 70)
print("INITIAL PREDICTIONS")
print("=" * 70)


for i in range(10):

    probability = predict_probability(
        model,
        X_train[i]
    )


    prediction = predict_class(
        probability
    )


    print(
        f"actual={y_train[i]} "
        f"probability={probability:.4f} "
        f"prediction={prediction}"
    )


print()


# ============================================================
# 27. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    X,
    y,
    learning_rate=0.01
):

    """
    One epoch means:

        process the training examples once.

    For every example:

        1. Forward
        2. Calculate loss
        3. Backward
        4. Update parameters
        5. Reset gradients

    """

    total_loss = 0.0


    # --------------------------------------------------------
    # Shuffle training examples
    # --------------------------------------------------------

    indices = list(
        range(len(X))
    )

    random.shuffle(indices)


    # --------------------------------------------------------
    # Process each example
    # --------------------------------------------------------

    for index in indices:

        # ----------------------------------------------------
        # Convert NumPy features → Value objects
        # ----------------------------------------------------

        inputs = numpy_to_values(
            X[index]
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logit = model(inputs)


        # ----------------------------------------------------
        # Sigmoid
        # ----------------------------------------------------

        probability = sigmoid(
            logit
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = binary_cross_entropy(
            probability,
            y[index]
        )


        # ----------------------------------------------------
        # Accumulate loss for reporting
        # ----------------------------------------------------

        total_loss += loss.data


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Gradient descent
        # ----------------------------------------------------

        for parameter in model.parameters():

            parameter.data -= (
                learning_rate
                * parameter.grad
            )


        # ----------------------------------------------------
        # Reset gradients
        # ----------------------------------------------------

        for parameter in model.parameters():

            parameter.grad = 0.0


    # --------------------------------------------------------
    # Average loss
    # --------------------------------------------------------

    average_loss = (
        total_loss
        /
        len(X)
    )


    return average_loss


# ============================================================
# 28. EVALUATION LOSS
# ============================================================

def calculate_loss(
    model,
    X,
    y
):

    """
    Calculate average BCE loss.

    IMPORTANT:

    We DO NOT call backward() here.

    Validation/test evaluation should not update the model.
    """

    total_loss = 0.0


    for index in range(len(X)):

        inputs = numpy_to_values(
            X[index]
        )


        logit = model(inputs)


        probability = sigmoid(
            logit
        )


        loss = binary_cross_entropy(
            probability,
            y[index]
        )


        total_loss += loss.data


    return (
        total_loss
        /
        len(X)
    )


# ============================================================
# 29. ACCURACY
# ============================================================

def calculate_accuracy(
    model,
    X,
    y,
    threshold=0.5
):

    """
    Accuracy:

        correct predictions
        /
        total predictions

    This is useful, but later we will also calculate:

        precision
        recall
        F1
        confusion matrix

    because spam classification has class imbalance.
    """

    correct = 0


    for index in range(len(X)):

        probability = predict_probability(
            model,
            X[index]
        )


        prediction = predict_class(
            probability,
            threshold
        )


        if prediction == y[index]:

            correct += 1


    return correct / len(X)


# ============================================================
# 30. TRAINING LOOP
# ============================================================

"""
Now we finally train.

Each epoch:

    training data
        ↓
    forward
        ↓
    loss
        ↓
    backward
        ↓
    gradient descent
        ↓
    updated weights

Then we evaluate validation data.

"""

epochs = 5

learning_rate = 0.01


print("=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(
    1,
    epochs + 1
):

    train_loss = train_one_epoch(
        model,
        X_train,
        y_train,
        learning_rate
    )


    validation_loss = calculate_loss(
        model,
        X_validation,
        y_validation
    )


    validation_accuracy = calculate_accuracy(
        model,
        X_validation,
        y_validation
    )


    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"validation_loss={validation_loss:.4f} | "
        f"validation_accuracy={validation_accuracy:.4f}"
    )


# ============================================================
# 31. FINAL TEST EVALUATION
# ============================================================

"""
IMPORTANT:

We have not used the test set for training.

We also have not used it for choosing the architecture.

Now we can perform a final evaluation.
"""

test_loss = calculate_loss(
    model,
    X_test,
    y_test
)


test_accuracy = calculate_accuracy(
    model,
    X_test,
    y_test
)


print()


print("=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Test accuracy: {test_accuracy:.4f}"
)

print()


# ============================================================
# 32. TRY REAL SMS MESSAGES
# ============================================================

"""
We can now type a completely new message.

IMPORTANT:

The new message must use:

    training vocabulary
    training IDF

We must NOT rebuild TF-IDF from the new message.
"""

def message_to_vector(message):

    tokens = tokenize(
        message
    )


    return tfidf_vector(
        tokens,
        word_to_index,
        idf
    )


def classify_message(
    message,
    threshold=0.5
):

    vector = message_to_vector(
        message
    )


    probability = predict_probability(
        model,
        vector
    )


    prediction = predict_class(
        probability,
        threshold
    )


    if prediction == 1:

        label = "SPAM"

    else:

        label = "HAM"


    return (
        label,
        probability
    )


# ============================================================
# 33. EXAMPLE REAL-TIME PREDICTIONS
# ============================================================

examples = [
    "Hey, are we still meeting tomorrow?",
    "Congratulations! You have won a free prize. Call now!",
    "Can you send me the project file?",
    "URGENT! Claim your FREE cash reward now!"
]


print("=" * 70)
print("NEW MESSAGE PREDICTIONS")
print("=" * 70)


for message in examples:

    label, probability = classify_message(
        message
    )


    print()
    print("Message:")
    print(message)

    print(
        f"Prediction: {label}"
    )

    print(
        f"P(spam): {probability:.4f}"
    )


print()


# ============================================================
# END
# ============================================================

print("=" * 70)
print("PHASE 9 COMPLETE")
print("=" * 70)

print()
print("We connected:")
print()
print("TF-IDF")
print("   ↓")
print("YOUR MLP")
print("   ↓")
print("LOGIT")
print("   ↓")
print("SIGMOID")
print("   ↓")
print("PROBABILITY")
print("   ↓")
print("BCE LOSS")
print("   ↓")
print("BACKPROPAGATION")
print("   ↓")
print("GRADIENT DESCENT")
