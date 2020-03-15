import tensorflow as tf
import numpy as np


tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, split=' ', char_level=False, oov_token=0, num_words=2)

text = "Das ist ein test"
sequence = tf.keras.preprocessing.text.text_to_word_sequence(text, lower=True, split=" ")
print(sequence)
tokenizer.fit_on_sequences(sequence)
# print(tokenizer)
test = "ist ein"
test = tf.keras.preprocessing.text.text_to_word_sequence(test, lower=True, split=" ")
print(tokenizer.sequences_to_matrix(test))
