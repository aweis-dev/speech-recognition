import tensorflow as tf

class db():
    def __init__(self):
        self.path = "C:/Machine Learning/speech-recognition/data/train/testing_list.txt"
        self.tensordata = self.filenames()

    def filenames(self):
        data = open(self.path, "r")
        length = sum(1 for l in data)
        data.close()
        tensordata = tf.Variable(tf.zeros([length, 2], tf.string))

        data = open(self.path, "r")
        i = 0
        for line in data:
            z = line.split(sep = '/')
            z[1] = z[1].rstrip('\n')
            tensordata[i].assign(z)
            i += 1

        # print(tensordata)

        # Pisser muss weg
        data.close()

        return tensordata

    def loadaudio(self):
        path = "C:/Machine Learning/speech-recognition/data/train/audio/" + self.tensordata[0, 0] \
               + "/" + self.tensordata[0, 1]
        with open(path, "rb") as kek:
            audio = tf.audio.decode_wav(kek)
            print(audio)

