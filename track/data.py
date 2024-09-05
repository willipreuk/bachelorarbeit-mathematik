from typing import Any
import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt

file_path = "data/Hhwayli.dat"


def read_data() -> tuple[ndarray[float, dtype[Any]], ndarray[float, dtype[Any]]]:
    x_vals = np.arange(0, 409.6, 0.05)

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [float(x) for x in data]

    return np.array(data), x_vals
    # return np.array(1/50 * (np.sin(x_vals))), x_vals


def read_data_interpolated() -> tuple[ndarray[float, dtype[Any]], ndarray[float, dtype[Any]]]:
    data, x_vals = read_data()
    x_vals_fine = np.arange(0, 409.6, 0.005)

    return np.interp(x_vals_fine, x_vals, data), x_vals_fine


def plot_data():
    data, x_vals = read_data()

    plt.figure()
    nn_x = np.linspace(0, 409.6, 10000)

    plt.plot(x_vals, data)
    plt.show()


def plot_freq(data, dx):

    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), d=dx)

    # Plot the FFT result
    plt.figure()
    plt.stem(fft_freq, np.abs(fft_result), 'b', markerfmt=" ", basefmt="-b")
    plt.title('FFT of Data')
    plt.xlim(left=0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()


if __name__ == '__main__':
    data, x_vals = read_data()
    # plot_freq(data, 0.05)
    plot_data()