import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv1D

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.d1 = Dense(1000, activation="relu")
        self.out = Dense(10)

    def call(self, x):
        x = self.d1(x)
        return self.out(x)

