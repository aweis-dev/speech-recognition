import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt

class db():
    def __init__(self):
        self.path = "../data/train/testing_list.txt"
        self.tensordata = self.filenames()

    def filenames(self):
        data = open(self.path, "r")
        length = sum(1 for l in data)
        data.close()
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
            path = "../data/train/audio/" + str(self.tensordata[index,0].numpy(), "utf-8") + "/" + str(self.tensordata[index,1].numpy(), "utf-8")
            fs, data = wavfile.read(path)
            audio_tensor[index].assign(data)
        return audio_tensor

    def getTrueValues(self, index_arr):
        index_len = len(index_arr)
        true_value_tensor = tf.Variable(tf.zeros([index_len], tf.string))
        for index in index_arr:
            true_value_tensor[index].assign(self.tensordata[index, 0])
        return true_value_tensor


