import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_transform as tft
from scipy.io import wavfile

class db():
    def __init__(self):
        self.path = "../data/train/testing_list.txt"
        self.length = 0
        self.tensordata = self.filenames()
        self.dict = {"Yes":0, "No":1, "Up":2, "Down":3, "Left":4, "Right":5, "On":6, "Off":7,
                     "Stop":8, "Go":9, "Zero":10, "One":11, "Two":12, "Three":13,
                     "Four":14, "Five":15, "Six":16, "Seven":17, "Eight":18, "Nine":19,
                     "Bed":20, "Bird":21, "Cat":22, "Dog":23, "Happy":24,
                     "House":25, "Marvin":26, "Sheila":27, "Tree":28, "Wow":29}

    def filenames(self):
        data = open(self.path, "r")
        length = sum(1 for l in data)
        data.close()
        self.length = length
        tensordata = tf.Variable(tf.zeros([length, 2], tf.string))
        data = open(self.path, "r")
        i = 0
        for line in data:
            line = str(line)
            z = line.split(sep = '/')
            z[1] = z[1].rstrip('\n')
            tensordata[i].assign(z)
            i += 1
        # print(tensordata)
        data.close()
        return tensordata

    def loadAudio(self, index_arr): # index_arr
        index_len = len(index_arr)
        audio_tensor = tf.Variable(tf.zeros([index_len, 16000], tf.int32))
        for index in index_arr:
            path = "../data/train/audio/" + str(self.tensordata[index,0].numpy(), "utf-8") + "/" \
                   + str(self.tensordata[index,1].numpy(), "utf-8")
            fs, data = wavfile.read(path)
            audio_tensor[index].assign(data)
        return audio_tensor

    def getTrueValues(self, index_arr):
        index_len = len(index_arr)
        true_value_tensor = tf.Variable(tf.zeros([index_len, 30], tf.float32))
        for index in index_arr:
            word_index = self.dict[self.tensordata[index, 0]]
            onehotencoding = tf.zeros([30])
            onehotencoding[word_index] = 1
            true_value_tensor[index].assign(onehotencoding)
        return true_value_tensor

    def normalizeWithMoments(self, x):
        offset = 0.0
        scale = 1.0
        epsilon = 1e-15
        mean, variance = tf.nn.moments(x, axes=[0, 1])
        x_normed = tf.nn.batch_normalization(x, mean, variance, offset, scale, epsilon)
        return x_normed

    def preprocessAudio(self, audio_tensor):
        data = tf.dtypes.cast(audio_tensor, dtype=tf.float64)
        signal = db.normalizeWithMoments(self, data)

        data = tf.dtypes.cast(signal, dtype=tf.complex64)
        fft_audio = tf.abs(tf.signal.fft(data))
        signal_fft = db.normalizeWithMoments(self, fft_audio)
        return signal, signal_fft


