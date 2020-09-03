import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from model import Model

USE_GPU = True

NUM_FILES = 200
MAX_FILE_LINES = 5
TEST_PERCENT = 20

IS_BIDIRECTIONAL = False
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.002
EMBEDDING_DIM = 64
RNN_UNITS = 2000

START_ID = 1
END_ID = 2
NUM_STRUCTURE_TOKENS = 37#15

residue_to_id = dict()  # Map residue tokens to IDs
id_to_residue = dict()  # Map IDs to residue tokens


def map_tokens():
    """Assign each residue token to a unique ID"""
    residue_to_id["<s>"] = START_ID
    id_to_residue[START_ID] = "<s>"

    residue_to_id["<e>"] = END_ID
    id_to_residue[END_ID] = "<e>"

    with open(f"data{os.sep}tokens.txt") as tokens_file:
        for token in tokens_file.readlines():
            token_id = len(residue_to_id) + 1
            residue_to_id[token] = token_id
            id_to_residue[token_id] = token


def load_data():
    """Returns 4 lists: sequences_train, sequences_test,
    structures_train, structures_test
    """
    sequences = []
    structures = []

    file_names = os.listdir(f"data{os.sep}seq")
    random.shuffle(file_names)
    files_added = 0
    for file_name in file_names:
        if files_added >= NUM_FILES:
            break
        
        try:
            # Open sequence file
            with open(
                f"data{os.sep}seq{os.sep}{file_name}"
            ) as seq_file:

                seq_file_lines = seq_file.readlines()
                if MAX_FILE_LINES != -1 and len(seq_file_lines) > MAX_FILE_LINES:
                    continue

                # Open structure file
                with open(
                    f"data{os.sep}struct{os.sep}{file_name}"
                ) as struct_file:
                    
                    # Structure IDs:
                    # 0: padding
                    # 1: start
                    # 2: end
                    # 3: no distinct secondary structure
                    # 4-13: helix class as specified at:
                    #     http://www.wwpdb.org/documentation/file-format-content/format33/sect5.html#HELIX
                    # 14: sheet
                    # has_secondary_structures = False
                    # structure = [START_ID]
                    # for line in struct_file.readlines():
                    #     structure_id = int(line) + 3
                    #     if structure_id != 3:
                    #         has_secondary_structures = True
                    #     structure.append(structure_id)
                    # if not has_secondary_structures:
                    #     continue  # Only use PDB files with helices/sheets
                    # structure.append(END_ID)
                    # structures.append(structure)
                    structure = [START_ID]
                    for line in seq_file_lines:
                        structure.append(residue_to_id[line])
                    structure.append(END_ID)
                    structures.append(structure)

                    sequence = [START_ID]
                    for line in seq_file_lines:
                        sequence.append(residue_to_id[line])
                    sequence.append(END_ID)
                    sequences.append(sequence)

                    files_added += 1

                    print(f"    {(files_added / NUM_FILES * 100):.2f}%")

        except:
            print("Invalid file name: " + file_name)

    return train_test_split(
        sequences, structures, test_size=TEST_PERCENT / 100
    )


def load_data_random():
    """Returns 4 lists: sequences_train, sequences_test,
    structures_train, structures_test, where each train
    list and each test list consists of the same set of random numbers
    """
    sequences = []
    structures = []

    for i in range(NUM_FILES):
        sequence = [
            random.randrange(3, len(residue_to_id) + 1, 1)
            for i in range(MAX_FILE_LINES)
        ]

        sequence.insert(0, START_ID)
        sequence.append(END_ID)

        sequences.append(sequence)
        structures.append(sequence.copy())

    return train_test_split(
        sequences, structures, test_size=TEST_PERCENT / 100
    )


def main():
    if not USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Mapping residues to token IDs...")
    map_tokens()

    print("Loading data...")
    sequences_train, sequences_test, structures_train, structures_test = \
        load_data_random()

    encoder = Encoder(
        RNN_UNITS,
        len(residue_to_id) + 1,
        EMBEDDING_DIM,
        BATCH_SIZE,
        IS_BIDIRECTIONAL
    )
    decoder = Decoder(
        RNN_UNITS,
        NUM_STRUCTURE_TOKENS,
        EMBEDDING_DIM
    )
    
    print("Training model...")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    model = Model(encoder, decoder, optimizer, loss_object)
    model.train(
        sequences_train, structures_train,
        BATCH_SIZE, EPOCHS, START_ID
    )

    print("Making predictions...")
    while True:
        prediction, input_seq, target_seq = model.predict_random(
            sequences_test, structures_test, START_ID, END_ID
        )
        print(f"Input:      {input_seq}")
        print(f"Target:     {target_seq}")
        print(f"Prediction: {prediction}")
        input("Press enter to continue.\n")


if __name__ == "__main__":
    main()
