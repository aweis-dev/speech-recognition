import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_transform as tft
from scipy.io import wavfile

class db():
    def __init__(self, colab=False):
        self.colab = colab
        if self.colab == False:
            self.path = "../data/train/list.txt"
        else:
            self.path = "../data/train/list.txt"
        self.length = 0
        self.tensordata = self.filenames()
        self.dict = {"yes":0, "no":1, "up":2, "down":3, "left":4, "right":5, "on":6, "off":7,
                     "stop":8, "go":9, "zero":10, "one":11, "two":12, "three":13,
                     "four":14, "five":15, "six":16, "seven":17, "eight":18, "nine":19,
                     "bed":20, "bird":21, "cat":22, "dog":23, "happy":24,
                     "house":25, "marvin":26, "sheila":27, "tree":28, "wow":29}

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
        index_arr = list(index_arr)
        index_len = len(index_arr)
        i = 0
        audio_tensor = tf.Variable(tf.zeros([index_len, 16000], dtype=tf.int16))
        for index in index_arr:
            if self.colab == False:
                path = "../data/train/audio/" + str(self.tensordata[index,0].numpy(), "utf-8") + "/" \
                    + str(self.tensordata[index,1].numpy(), "utf-8")
            else:
                path = "../data/train/audio/" + str(self.tensordata[index,0].numpy(), "utf-8") + "/" \
                    + str(self.tensordata[index,1].numpy(), "utf-8")
            fs, data = wavfile.read(path)
            if len(data) > 16000:
                data = data[:16000]
            temp = tf.Variable(tf.zeros(16000, dtype=tf.int16))
            temp[0:len(data)].assign(tf.Variable(data))
            data = temp
            audio_tensor[i].assign(data)
            i += 1
        # print(audio_tensor)
        return audio_tensor

    def getTrueValues(self, index_arr):
        index_len = len(index_arr)
        true_value_tensor = tf.Variable(tf.zeros([index_len, 30], dtype=tf.float32))
        i = 0
        for index in index_arr:
            word_index = self.dict[str(self.tensordata[index, 0].numpy(), "utf-8")]
            # print(word_index)
            onehotencoding = tf.Variable(tf.zeros([30]), dtype=tf.float32)
            onehotencoding[word_index].assign(1.)
            true_value_tensor[i].assign(onehotencoding)
            i += 1
        # print(true_value_tensor)
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


