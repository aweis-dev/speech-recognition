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

    def loadaudio(self):
        # print(str(self.tensordata[0,0].numpy(), "utf-8"))
        path = "../data/train/audio/" + str(self.tensordata[0,0].numpy(), "utf-8") + "/" + str(self.tensordata[0,1].numpy(), "utf-8")
        # print(path)
        fs, data = wavfile.read(path)
        print(fs)
        plt.plot(data)
        plt.show()
        # audio = tf.audio.decode_wav(kek)
        # print(audio)

