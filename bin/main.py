import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow_transform as tft
import database
import model


def make_plot(audio, audio_fft):
    plt.subplot(2,1,1)
    plt.title("Signal and FFT of signal")
    plt.plot(audio)
    plt.xlabel("Time")
    plt.ylabel("Signal")

    plt.subplot(2,1,2)
    plt.plot(audio_fft)
    plt.xlabel("Frequency")
    plt.ylabel("FFT of signal")
    plt.show()

    return None



if __name__ == "__main__":
    model = model.Model()
    model.train(500, 100)
