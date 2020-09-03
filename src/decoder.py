import tensorflow as tf

from attention import Attention


class Decoder(tf.keras.Model):
    def __init__(self, units, vocab_size, embedding_dim):
        super().__init__()

        self.embedding_layer = tf.keras.layers.Embedding(
            vocab_size, embedding_dim
        )

        self.attention_layer = Attention(units)
        self.gru_layer = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.prediction_dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, prev_hidden_state, encoder_output):
        context_vector, attention_weights = \
            self.attention_layer(prev_hidden_state, encoder_output)
        
        # (batch size, 1, embedding_dim)
        x = self.embedding_layer(x)

        # (batch size, 1, units)
        expanded_context_vector = tf.expand_dims(context_vector, 1)
        
        # (batch_size, 1, units + embedding_dim)
        x = tf.concat([expanded_context_vector, x], axis=-1)

        # output: (batch size, timesteps (1), units)
        output, state = self.gru_layer(x)

        # (batch size * timesteps (1), units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # (batch size, vocab_size)
        predictions = self.prediction_dense(output)

        return predictions, state, attention_weights
