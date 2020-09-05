import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, x, y, x_vocabulary=None, y_vocabulary=None, test_size=0.2):
        """Initialize dataset from x and y (lists of token lists) and
        generate vocabularies from vocabulary lists, if provided. Otherwise,
        generate vocabularies from x and y"""
        self.x_to_id = dict()
        self.id_to_x = dict()
        
        self.y_to_id = dict()
        self.id_to_y = dict()

        padding_token = "<PAD>"
        self.padding_token_id = 0
        self.__add_x_token(padding_token)
        self.__add_y_token(padding_token)

        start_token = "<START>"
        self.start_token_id = 1
        self.__add_x_token(start_token)
        self.__add_y_token(start_token)

        end_token = "<END>"
        self.end_token_id = 2
        self.__add_x_token(end_token)
        self.__add_y_token(end_token)

        # Generate vocabulary
        if x_vocabulary is None:
            for sequence in x:
                for token in sequence:
                    if token not in self.x_to_id:
                        self.__add_x_token(token)
        else:
            for token in x_vocabulary:
                if token not in self.x_to_id:
                    self.__add_x_token(token)

        if y_vocabulary is None:
            for sequence in y:
                for token in sequence:
                    if token not in self.y_to_id:
                        self.__add_y_token(token)
        else:
            for token in y_vocabulary:
                if token not in self.y_to_id:
                    self.__add_y_token(token)

        # Get maximum length of x and y sequences
        max_x_len = 0
        for sequence in x:
            if len(sequence) > max_x_len:
                max_x_len = len(sequence)
        max_x_len += 2  # Add 2 for start/end tokens

        max_y_len = 0
        for sequence in y:
            if len(sequence) > max_y_len:
                max_y_len = len(sequence)
        max_y_len += 2  # Add 2 for start/end tokens

        # Replace tokens with IDs, add zero padding
        x_processed = np.zeros((len(x), max_x_len))
        for i, sequence in enumerate(x):
            x_processed[i, 0] = self.start_token_id
            j = 1
            for token in sequence:
                x_processed[i, j] = self.x_to_id[token]
                j += 1
            x_processed[i, j] = self.end_token_id

        y_processed = np.zeros((len(y), max_y_len))
        for i, sequence in enumerate(y):
            y_processed[i, 0] = self.start_token_id
            j = 1
            for token in sequence:
                y_processed[i, j] = self.y_to_id[token]
                j += 1
            y_processed[i, j] = self.end_token_id

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x_processed, y_processed, test_size=test_size)

    def __add_x_token(self, token):
        token_id = len(self.x_to_id)
        self.x_to_id[token] = token_id
        self.id_to_x[token_id] = token

    def __add_y_token(self, token):
        token_id = len(self.y_to_id)
        self.y_to_id[token] = token_id
        self.id_to_y[token_id] = token

    def process_input_sequence(self, sequence, accept_missing_vocab=True):
        """Returns a preprocessed sequence (numpy array of tokens)"""
        processed_sequence = [self.start_token_id]
        for i, token in enumerate(sequence):
            if token in self.x_to_id:
                processed_sequence.append(self.x_to_id[token])
            else:
                if accept_missing_vocab:
                    print(f"Token \"{token}\" not in dataset's vocabulary.")
                else:
                    raise ValueError(
                        f"Token \"{token}\" not in dataset's vocabulary."
                    )
        processed_sequence.append(self.end_token_id)
        return np.asarray(processed_sequence)

    def get_x_vocab_size(self):
        return len(self.x_to_id)

    def get_y_vocab_size(self):
        return len(self.y_to_id)
