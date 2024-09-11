from typing import Any
import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt

file_path = "data/Hhwayli.dat"

t_end = 10

def read_data() -> tuple[ndarray[float, dtype[Any]], ndarray[float, dtype[Any]]]:
    x_vals = np.arange(0, t_end, 0.089993)
    # x_vals_test = np.arange(0, 10, 0.01)

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [float(x) for x in data]

    data_values = np.array(data[:len(x_vals)])

    print("Shape of data: ", data_values.shape)

    return data_values, x_vals
    # return np.array(1/10 * (np.sin(x_vals_test)*0.4)), x_vals_test


def read_data_interpolated() -> tuple[ndarray[float, dtype[Any]], ndarray[float, dtype[Any]]]:
    data, x_vals = read_data()
    x_vals_fine = np.arange(0, t_end, 0.0001)

    data_fine = np.interp(x_vals_fine, x_vals, data)
    print("Shape of data: ", data_fine.shape)

    return data_fine, x_vals_fine


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
