import os
import random

import numpy as np
import tensorflow as tf

from dataset import Dataset
from encoder import Encoder
from decoder import Decoder
from model import Model

TEST_PERCENT = 20
IS_BIDIRECTIONAL = False
BATCH_SIZE = 16
EPOCHS = 120
LEARNING_RATE = 0.001
EMBEDDING_DIM = 15
ENCODER_UNITS = 45
DECODER_UNITS = ENCODER_UNITS * 2 if IS_BIDIRECTIONAL else ENCODER_UNITS

NUM_SEQUENCES = 1500
MIN_TRAIN_TERM = 0
MAX_TRAIN_TERM = 99
MIN_TEST_TERM = 0
MAX_TEST_TERM = 99
MAX_PREDICTION_LEN = 10


def gen_math_problem(min_term, max_term):
    """Generates a random addition or subtraction problem and
    returns two lists of problem and solution tokens"""
    term_1 = random.randint(min_term, max_term)
    term_2 = random.randint(min_term, max_term)
    add_terms = random.randint(0, 1)
    if add_terms:
        solution = term_1 + term_2
    else:
        solution = term_1 - term_2
    operation = "+" if add_terms else "-"

    problem = list(str(term_1)) + [operation] + list(str(term_2))
    solution = list(str(solution))

    return problem, solution


def load_data(num_sequences):
    """Returns sequence lists x and y, containing problems and
    solutions of random arithmetic problems"""
    x_vocabulary = [str(i) for i in range(10)]
    x_vocabulary.append("+")
    x_vocabulary.append("-")
    random.shuffle(x_vocabulary)

    y_vocabulary = [str(i) for i in range(10)]
    y_vocabulary.append("-")
    random.shuffle(y_vocabulary)

    x = []
    y = []

    for i in range(num_sequences):
        problem, solution = gen_math_problem(MIN_TRAIN_TERM, MAX_TRAIN_TERM)
        x.append(problem)
        y.append(solution)

    return x, y, x_vocabulary, y_vocabulary


def main():
    print("Loading data...")
    x, y, x_vocabulary, y_vocabulary = \
        load_data(NUM_SEQUENCES)
    dataset = Dataset(
        x, y,
        x_vocabulary=x_vocabulary, y_vocabulary=y_vocabulary
    )

    encoder = Encoder(
        ENCODER_UNITS,
        dataset.get_x_vocab_size(),
        EMBEDDING_DIM,
        BATCH_SIZE,
        IS_BIDIRECTIONAL
    )
    decoder = Decoder(
        DECODER_UNITS,
        dataset.get_y_vocab_size(),
        EMBEDDING_DIM
    )
    
    print("Training model...")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    model = Model(encoder, decoder, dataset, optimizer, loss_object)
    model.train(BATCH_SIZE, BATCH_SIZE, EPOCHS)

    print("Making predictions...")
    while True:
        problem, solution = gen_math_problem(MIN_TEST_TERM, MAX_TEST_TERM)
        prediction = model.predict(problem, MAX_PREDICTION_LEN)
        print(f"Input:      {''.join(problem)}")
        print(f"Target:     {''.join(solution)}")
        print(f"Prediction: {''.join(prediction)}")
        input("Press enter to continue.\n")


if __name__ == "__main__":
    main()
