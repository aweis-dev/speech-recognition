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

    def train(self):
        return None

    def test(self):
        return None
