import random

import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, encoder, decoder, dataset, optimizer, loss_object):
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_object = loss_object

    def __gen_batch(self, x, y, batch_size, shuffle=False):
        """Generate batches for model training"""
        if shuffle:
            data_pairs = list(zip(x, y))
            random.shuffle(data_pairs)
            x, y = zip(*data_pairs)
            x = np.array(x)
            y = np.array(y)

        # For each batch of data
        for i in range(0, len(x) - batch_size + 1, batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    def __get_loss(self, actual_batch, prediction_batch):
        """Get the overall loss for a batch of actual and predicted values
        
        actual_batch shape: (batch size,),
        prediction_batch shape: (batch size, vocab size)
        """
        # Ignore padding
        # e.g. [[5], [6], [0]] -> [[True], [True], [False]]
        mask = tf.math.logical_not(tf.math.equal(actual_batch, 0))
        
        loss = self.loss_object(actual_batch, prediction_batch)
        
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_mean(loss)

    def __evaluate(self, input_batch, target_batch, batch_size):
        total_loss = 0
        
        encoder_initial_state = tf.zeros((batch_size, self.encoder.units))
        encoder_output, encoder_state = self.encoder(
            input_batch, encoder_initial_state
        )
        decoder_state = encoder_state

        # Begin each input in batch with start ID
        decoder_input = tf.expand_dims(
            [self.dataset.start_token_id] * batch_size, 1
        )

        # For each token (excluding start)
        for i in range(1, target_batch.shape[1]):
            predictions, decoder_state, _ = self.decoder(
                decoder_input, decoder_state, encoder_output
            )

            total_loss += self.__get_loss(
                target_batch[:, i], predictions
            )

            # Update decoder_input with current token
            decoder_input = tf.expand_dims(target_batch[:, i], 1)

        batch_loss = total_loss / target_batch.shape[1]

        return batch_loss

    @tf.function
    def __train_step(self, input_batch, target_batch, batch_size):
        total_loss = 0
        
        with tf.GradientTape() as tape:
            encoder_output, encoder_state = self.encoder(input_batch)
            
            decoder_state = encoder_state

            # Begin each input in batch with start ID
            decoder_input = tf.expand_dims(
                [self.dataset.start_token_id] * batch_size, 1
            )

            # For each token (excluding start)
            for i in range(1, target_batch.shape[1]):
                predictions, decoder_state, _ = self.decoder(
                    decoder_input, decoder_state, encoder_output
                )

                total_loss += self.__get_loss(
                    target_batch[:, i], predictions
                )

                # Update decoder_input with current token
                decoder_input = tf.expand_dims(target_batch[:, i], 1)

        batch_loss = total_loss / target_batch.shape[1]

        variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self, batch_size, val_batch_size, epochs):
        STEPS_PER_EPOCH = len(self.dataset.x_train) // batch_size

        for epoch in range(epochs):
            print(f"Training epoch {epoch}...")

            epoch_loss = 0

            batch_generator = self.__gen_batch(
                self.dataset.x_train,
                self.dataset.y_train,
                batch_size
            )
            
            # For each batch of data
            for step_num, (input_batch, target_batch) in enumerate(batch_generator):
                batch_loss = self.__train_step(
                    input_batch, target_batch, batch_size
                )

                epoch_loss += batch_loss
            
            VAL_STEPS = len(self.dataset.x_test) // val_batch_size

            total_val_loss = 0

            val_batch_generator = self.__gen_batch(
                self.dataset.x_test,
                self.dataset.y_test,
                val_batch_size,
                shuffle=True
            )
            for (val_input_batch, val_target_batch) in val_batch_generator:
                batch_loss = self.__evaluate(
                    val_input_batch, val_target_batch, val_batch_size
                )
                total_val_loss += batch_loss
            
            print(f"    Loss: {(epoch_loss / STEPS_PER_EPOCH):.2f}")
            print(f"    Validation loss: {(total_val_loss / VAL_STEPS):.2f}")

    def predict(self, x, max_len):
        """Returns a predicted output sequence given sequence x"""
        x = np.array([self.dataset.process_input_sequence(x)])

        encoder_initial_state = tf.zeros((1, self.encoder.units))
        encoder_output, encoder_state = self.encoder(
            x, encoder_initial_state
        )

        decoder_state = encoder_state
        decoder_input = tf.constant([[self.dataset.start_token_id]])

        prediction = []

        # Generate prediction until end token or max_len reached
        for i in range(max_len):
            next_id_prediction, decoder_state, _ = self.decoder(
                decoder_input, decoder_state, encoder_output 
            )
            predicted_id = tf.argmax(next_id_prediction[0]).numpy()
            if predicted_id == self.dataset.end_token_id:
                break

            prediction.append(self.dataset.id_to_y[predicted_id])
            decoder_input = tf.constant([[predicted_id]])

        return prediction
