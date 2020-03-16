import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas
import tfnetwork
import database
import os


class Model():
    def __init__(self):
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # uses one hot encoding! 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-7)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)
        self.db = database.db()
        self.network = tfnetwork.Network()

    def train(self, epoch, batchsize):
        # checkpoint_path = "params/cp.ckpt"
        # checkpoit_dir = os.path.dirname(checkpoint_path)
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        for i in range(epoch):
            rndmbatch = np.random.random_integers(0, high=self.db.length, size=batchsize)
            audio_tensor = self.db.loadAudio(rndmbatch)
            true_values = self.db.getTrueValues(rndmbatch)
            signal, signal_fft = self.db.preprocessAudio(audio_tensor)
            with tf.GradientTape() as tape:
                output = self.network(signal, signal_fft)
                loss = self.loss_function(true_values, output)
                # print(db.dict[output])
                # print(output[0])
                # print(true_values[0])
            train_loss(loss)
            train_accuracy(true_values, output)
            gradients = tape.gradient(loss, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
            print("Epoch:", i, "Accuracy: ",  float(train_accuracy.result().numpy())*100, "%")
            train_loss.reset_states()
            train_accuracy.reset_states()

        # To Do
        return None

    def test(self):
        return None
