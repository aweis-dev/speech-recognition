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
            # print(line)
            z = line.split(sep = '/')
            # z[0] = z[0].decode('utf-8')
            z[1] = z[1].rstrip('\n')
            # print(z)
            tensordata[i].assign(z)
            i += 1

        # print(tensordata)

        # Pisser muss weg
        data.close()

        return tensordata

    def loadaudio(self, index_arr): # index_arr
        index_len = len(index_arr)
        audio_tensor = tf.Variable(tf.zeros([index_len, 16000], tf.int32))
        for index in index_arr:
            path = "../data/train/audio/" + str(self.tensordata[index,0].numpy(), "utf-8") + "/" + str(self.tensordata[index,1].numpy(), "utf-8")
            fs, data = wavfile.read(path)
            audio_tensor[index].assign(data)
        # print(fs)
        # audio_tensor = tf.Variable(data)
        print(audio_tensor)
        return audio_tensor

