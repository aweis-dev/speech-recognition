import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPool1D

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        tf.keras.backend.set_floatx("float64")
        self.cnn1_layer1 = Conv1D(2, 5, activation="relu")
        self.cnn1_pool1  = MaxPool1D(pool_size=2, padding="same")
        self.cnn1_layer2 = Conv1D(3, 2, activation="relu")
        self.cnn1_pool2 = MaxPool1D(pool_size=2, padding="same")
        self.cnn1_layer3 = Conv1D(10, 2, activation="relu")
        # self.cnn1_pool3 = MaxPool1D(pool_size=2, padding="same")
        self.cnn1_layer4 = Conv1D(7, 2, activation="relu")
        # self.cnn1_pool4 = MaxPool1D(pool_size=2, padding="same")
        self.cnn1_layer5 = Conv1D(5, 2, activation="relu")
        # self.cnn1_pool5 = MaxPool1D(pool_size=2, padding="same")

        self.cnn2_layer1 = Conv1D(2, 5, activation="relu")
        self.cnn2_pool1 = MaxPool1D(pool_size=2, padding="same")
        self.cnn2_layer2 = Conv1D(3, 2, activation="relu")
        self.cnn2_pool2 = MaxPool1D(pool_size=2, padding="same")
        self.cnn2_layer3 = Conv1D(10, 2, activation="relu")
        # self.cnn2_pool3 = MaxPool1D(pool_size=2, padding="same")
        self.cnn2_layer4 = Conv1D(7, 2, activation="relu")
        # self.cnn2_pool4 = MaxPool1D(pool_size=2, padding="same")
        self.cnn2_layer5 = Conv1D(5, 2, activation="relu")
        # self.cnn2_pool5 = MaxPool1D(pool_size=2, padding="same")

        # Fully connected NN
        self.d1 = Dense(1000, activation="softplus")
        self.d2 = Dense(500, activation="softplus")
        self.d3 = Dense(100, activation="softplus")
        self.d4 = Dense(50, activation="softplus")

        # flatten
        self.flatten1 = Flatten(data_format="channels_last")
        self.flatten2 = Flatten(data_format="channels_last")

        # Probability distribution
        self.out = Dense(30)

    # x = signal, y = fast & the furious transformation
    def call(self, x, y):
        x_shape = tf.shape(x).numpy()
        y_shape = tf.shape(y).numpy()
        x = tf.reshape(x, [x_shape[0], x_shape[1], 1])
        y = tf.reshape(y, [y_shape[0], y_shape[1], 1])

        x = self.cnn1_pool1(self.cnn1_layer1(x))
        x = self.cnn1_pool2(self.cnn1_layer2(x))
        x = self.cnn1_layer3(x)
        x = self.cnn1_layer4(x)
        x = self.cnn1_layer5(x)

        y = self.cnn2_pool1(self.cnn2_layer1(y))
        y = self.cnn2_pool2(self.cnn2_layer2(y))
        y = self.cnn2_layer3(y)
        y = self.cnn2_layer4(y)
        y = self.cnn2_layer5(y)

        x = self.flatten1(x)
        y = self.flatten2(y)
        z = tf.keras.layers.concatenate([x, y], axis=-1)

        result = self.d1(z)
        result = self.d2(result)
        result = self.d3(result)
        result = self.d4(result)
        result = self.out(result)

        return result



