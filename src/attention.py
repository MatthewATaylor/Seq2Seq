import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.hidden_state_dense = tf.keras.layers.Dense(units)
        self.encoder_output_dense = tf.keras.layers.Dense(units)
        self.score_dense = tf.keras.layers.Dense(1)

    def call(self, prev_hidden_state, encoder_output_seq):
        # prev_hidden_state: (batch size, units) 
        # prev_state_with_time_axis: (batch size, 1, units)
        prev_state_with_time_axis = tf.expand_dims(prev_hidden_state, 1)

        # (batch size, max sequence length, 1)
        score = self.score_dense(
            # (batch size, max sequence length, units)
            tf.nn.tanh(
                # (batch size, 1, units)
                self.hidden_state_dense(prev_state_with_time_axis) + \
                    # (batch size, max sequence length, units)
                    self.encoder_output_dense(encoder_output_seq)
            )
        )

        # (batch size, max sequence length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # (batch size, max sequence length, units)
        context_vector = attention_weights * encoder_output_seq
        
        # Add together corresponding hidden state values across timesteps
        # (batch size, units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
