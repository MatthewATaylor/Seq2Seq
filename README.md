# Seq2Seq
A bidirectional sequence-to-sequence RNN with attention

## Dependencies
* Tensorflow
* NumPy
* scikit-learn

## Model Description
The model's RNN utilizes gated recurrent units (GRUs) and can be trained in unidirectional or bidirectional modes. After input data is passed through an encoder RNN, a Bahdanau Attention mechanism is used to generate a context vector from weighted tokens. This is then applied to the decoder RNN to produce an output prediction.
