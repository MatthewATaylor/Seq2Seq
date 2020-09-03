import random

import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, encoder, decoder, optimizer, loss_object):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_object = loss_object

    def __max_list_len(self, lists):
        """Return the length of the list inside lists with the
        greatest length
        """
        max_len = -1
        for inner_list in lists:
            if len(inner_list) > max_len:
                max_len = len(inner_list)
        return max_len

    def __gen_batch(self, x, y, batch_size, shuffle=False):
        """Generate batches for model training"""
        input_size = self.__max_list_len(x)
        target_size = self.__max_list_len(y)

        # Shuffle data
        if shuffle:
            data_pairs = list(zip(x, y))
            random.shuffle(data_pairs)
            x, y = zip(*data_pairs)

        # For each batch of data
        for i in range(0, len(x), batch_size):
            input_batch = np.zeros((batch_size, input_size))
            target_batch = np.zeros((batch_size, target_size))

            # For each x/y list in batch
            for j in range(len(x[i:i + batch_size])):
                # For each residue
                for k, residue in enumerate(x[i]):
                    input_batch[j, k] = residue
                
                # For each structure
                for k, structure in enumerate(y[i]):
                    target_batch[j, k] = structure
                    
            yield (input_batch, target_batch)

    def __get_loss(self, actual_batch, prediction_batch):
        """Get the overall loss for a batch of actual and predicted values
        
        actual_batch shape: (batch size,),
        prediction_batch shape: (batch size, vocab size)
        """
        # Ignore padding
        # e.g. [[5], [6], [0]] -> [[True], [True], [False]]
        # mask = tf.math.logical_not(tf.math.equal(actual_batch, 0))
        
        loss = self.loss_object(actual_batch, prediction_batch)
        
        # mask = tf.cast(mask, dtype=loss.dtype)
        # loss *= mask
        return tf.reduce_mean(loss)

    def __get_accuracy(self, actual_batch, prediction_batch):
        """Get the overall accuracy for a batch of actual and predicted values
        
        actual_batch shape: (batch size,),
        prediction_batch shape: (batch size, vocab size)
        """
        # predicted_values = tf.math.argmax(prediction_batch)
        # actual_batch_int = tf.cast(actual_batch, predicted_values.dtype)
        
        # correct_values = 0
        # for i in range(actual_batch.shape[0]):
        #     if actual_batch_int[i] == predicted_values[i]:
        #         correct_values += 1
        
        # return correct_values / actual_batch.shape[0]
        return 0

    @tf.function
    def __train_step(
        self, input_batch, target_batch, start_token_id, batch_size
    ):
        loss = 0
        accuracy = 0
        
        with tf.GradientTape() as tape:
            encoder_output, encoder_state = self.encoder(input_batch)
            
            # Begin each input in batch with start ID
            decoder_input = tf.expand_dims(
                [start_token_id] * batch_size, 1
            )

            # For each token (excluding start)
            for i in range(1, target_batch.shape[1]):
                predictions, decoder_state, _ = self.decoder(
                    decoder_input, encoder_state, encoder_output
                )

                loss += self.__get_loss(
                    target_batch[:, i], predictions
                )
                accuracy += self.__get_accuracy(target_batch[:, i], predictions)

                # Update targets with current token
                decoder_input = tf.expand_dims(target_batch[:, i], 1)

        batch_loss = loss / int(target_batch.shape[1])
        batch_accuracy = accuracy / int(target_batch.shape[1])

        variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss, batch_accuracy

    def train(self, x, y, batch_size, epochs, start_token_id):
        steps_per_epoch = len(x) // batch_size

        for epoch in range(epochs):
            loss = 0
            accuracy = 0

            # For each batch of data
            for step_num, (input_batch, target_batch) in enumerate(
                self.__gen_batch(x, y, batch_size)
            ):
                batch_loss, batch_accuracy = self.__train_step(
                    input_batch, target_batch, start_token_id, batch_size
                )

                loss += batch_loss
                accuracy += batch_accuracy
                
                print(f"Epoch {epoch}, step {step_num} loss: {batch_loss:.2f}")
            
            print(f"Epoch {epoch}:")
            print(f"    Loss: {(loss / steps_per_epoch):.2f}")
            print(f"    Accuracy: {(accuracy / steps_per_epoch):.2f}")

    def predict_random(self, x, y, start_token_id, end_token_id):
        """Predicts the structures for a sequence randomly selected
        from x
        
        Returns (prediction, input, target)
        """
        prediction = []
        
        input_sequence, target_sequence = next(self.__gen_batch(
            x, y, batch_size=1, shuffle=True
        ))

        encoder_initial_state = tf.zeros((1, self.encoder.units))
        encoder_output, encoder_state = self.encoder(
            input_sequence, encoder_initial_state
        )

        decoder_input = tf.Variable([[start_token_id]])

        for i in range(input_sequence.shape[1]):
            next_id_prediction, decoder_state, _ = self.decoder(
                decoder_input, encoder_state, encoder_output
            )
            predicted_id = tf.argmax(next_id_prediction[0]).numpy()
            prediction.append(predicted_id)

            if predicted_id == end_token_id:
                break

            decoder_input = tf.Variable([[predicted_id]])
        
        return \
            prediction, \
            input_sequence[0].astype(int).tolist(), \
            target_sequence[0].astype(int).tolist()

