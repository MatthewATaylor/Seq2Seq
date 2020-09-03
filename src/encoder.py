import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(
        self, units, vocab_size, embedding_dim, batch_size,
        is_bidirectional
    ):
        super().__init__()

        self.units = units
        self.batch_size = batch_size
        self.is_bidirectional = is_bidirectional
        self.embedding_layer = tf.keras.layers.Embedding(
            vocab_size, embedding_dim
        )

        gru_layer = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        if is_bidirectional:
            self.gru_layer = tf.keras.layers.Bidirectional(
                gru_layer
            )
        else:
            self.gru_layer = gru_layer


    def call(self, x, initial_state=None):
        if initial_state is None:
            initial_state = tf.zeros((self.batch_size, self.units))

        # (batch size, max sequence length, embedding_dim)
        x = self.embedding_layer(x)

        if self.is_bidirectional:
            output, state_forward, state_backward = self.gru_layer(
                x, initial_state=[initial_state, tf.identity(initial_state)]
            )
            state = tf.concat(
                [state_forward, state_backward],
                axis=-1
            )
        else:
            output, state = self.gru_layer(
                x, initial_state=initial_state
            )
        
        return output, state
