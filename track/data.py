import numpy as np
import matplotlib.pyplot as plt
from config import Config, Excitations


def _simulated_excitation(t):
    data_l = 0.1 * np.sin(2 * np.pi * (t + 0.4))
    data_r = 0.1 * np.sin(2 * np.pi * (t + 0.1))

    return data_l, data_r


def read_data():
    x_vals = np.arange(0, Config.t_end, Config.delta_t)

    if Config.excitation == Excitations.DATA_SPLINE or Config.excitation == Excitations.DATA_NEURAL_NETWORK:
        with open(Config.data_r_path, "r") as file:
            data_r = file.readlines()

        data_r = [float(x) for x in data_r]
        data_r = np.array(data_r[:len(x_vals)])

        with open(Config.data_l_path, "r") as file:
            data_l = file.readlines()

        data_l = [float(x) for x in data_l]
        data_l = np.array(data_l[:len(x_vals)])

    else:
        data_l, data_r = _simulated_excitation(x_vals)

    return data_l, data_r, x_vals


def _plot_data():
    data_l, data_r, x_vals = read_data()

    plt.figure()

    plt.plot(x_vals, data_l, label="Left")
    plt.plot(x_vals, data_r, label="Right")

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Data")


def _plot_freq():
    data_l, data_r, x_vals = read_data()

    fft_result_l = np.fft.fft(data_l)
    fft_freq_l = np.fft.fftfreq(len(data_r), d=data_l[1] - data_l[0])

    fft_result_r = np.fft.fft(data_r)
    fft_freq_r = np.fft.fftfreq(len(data_r), d=data_r[1] - data_r[0])

    plt.figure()

    plt.stem(fft_freq_l, np.abs(fft_result_l), 'b', markerfmt=" ", basefmt="-b")
    plt.stem(fft_freq_r, np.abs(fft_result_r), 'r', markerfmt=" ", basefmt="-b")

    plt.title('FFT of Data')
    plt.xlim(left=0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')


if __name__ == '__main__':
    _plot_data()
    _plot_freq()

    plt.show()
