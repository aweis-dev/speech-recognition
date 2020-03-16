import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas
import tfnetwork
import database


class Model():
    def __init__(self):
        self.loss_function = tf.keras.losses.CategoricalCrossentropy() # uses one hot encoding! 
        self.optimizer = tf.keras.optimizers.Adam()
        self.db = database.db()
        self.network = tfnetwork.Network()

    def train(self, epoch, batchsize):
        for i in range(epoch):
            rndmbatch = np.random.random_integers(0, high=self.db.length, size=batchsize)
            audio_tensor = self.db.loadAudio(rndmbatch)
            true_values = self.db.getTrueValues(rndmbatch)
            signal, signal_fft = self.db.preprocessAudio(audio_tensor)
            with tf.GradientTape() as tape:
                output = self.network(signal, signal_fft)
                loss = self.loss_function(output, true_values)
            gradients = tape.gradient(loss, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        # To Do
        return None

    def test(self):
        return None
