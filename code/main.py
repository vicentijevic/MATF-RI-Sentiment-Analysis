import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


def main():

    df = pd.read_csv('../data/SerbMR-2C.csv')
    df["class-att"] = df["class-att"].map({"NEGATIVE": 0, "POSITIVE": 1})
    print(df)

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    print(df)
    print(df.shape)

    # Creating a TensorFlow dataset from data frame
    target = df.pop('class-att')
    ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    for ex in ds_raw.take(3):
        tf.print(ex[0].numpy()[0][:50], ex[1])

    # Split into training, testing and validaton datasets
    tf.random.set_seed(1)
    ds_raw = ds_raw.shuffle(1682, reshuffle_each_iteration=False)
    ds_raw_test = ds_raw.take(800)
    ds_raw_train_valid = ds_raw.skip(800)
    ds_raw_train = ds_raw_train_valid.take(700)
    ds_raw_valid = ds_raw_train_valid.skip(700)

    # Tokenization
    tokenizer = tfds.deprecated.text.Tokenizer()
    token_counts = Counter()

    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        token_counts.update(tokens)

    print('Vocab-size:', len(token_counts))

    # Unique word to unique integer
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
    example_str = 'Baš je strašan film'
    print(encoder.encode(example_str))

    def encode(text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)

    tf.random.set_seed(1)
    for example in ds_train.shuffle(100).take(5):
        print('Sequence length:', example[0].shape)

    train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
    valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
    test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

    embedding_dim = 20
    vocab_size = len(token_counts) + 2

    tf.random.set_seed(1)

    bi_lstm_model = tf.keras.Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed-layer'),
        Bidirectional(LSTM(64, name='lstm-layer'), name='bidir-lstm'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    bi_lstm_model.summary()

    bi_lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = bi_lstm_model.fit(
        train_data,
        validation_data=valid_data,
        epochs=10
    )

    test_results = bi_lstm_model.evaluate(test_data)
    print('Test Acc.: {:.2f}%'.format(test_results[1]*100))


if __name__ == "__main__":
    main()
