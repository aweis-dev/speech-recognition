import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow_transform as tft
import database


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
    db = database.db()
    index_arr = [0,1,2]
    audio_tensor = db.loadAudio(index_arr)
    true_values = db.getTrueValues(index_arr)
    # print(audio_tensor)
    # print(true_values)
    signal, signal_fft = db.preprocessAudio(audio_tensor)
    signal = signal.numpy()
    signal_fft = signal_fft.numpy()
    make_plot(signal[0], signal_fft[0])
