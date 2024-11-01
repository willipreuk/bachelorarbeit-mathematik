import numpy as np
import matplotlib.pyplot as plt
import config


def simulated_excitation(t):
    """
    Simulated excitations for the left and right track.
    :param t:
    :return: tuple of left and right track data
    """

    # excitations from the paper https://doi.org/10.1007/978-3-642-01356-0_20
    # data_l = 0.1 * np.sin(2 * np.pi * (t + 0.4))
    # data_r = 0.1 * np.sin(2 * np.pi * (t + 0.1))

    def f(x):
        return 0.01 * (np.sin(0.2 * np.pi * x) + np.sin(0.4 * np.pi * x) + np.sin(0.75 * np.pi * x) + np.sin(1 * np.pi * x) + np.sin(1.5 * np.pi * x))

    return f(t), f(t + config.phase_shift)


def simulated_diff_excitation(t):
    """
    Simulated differential excitations for the left and right track.
    :param t:
    :return: tuple of left and right diff track data
    """

    # diff excitation from the paper https://doi.org/10.1007/978-3-642-01356-0_20
    # data_l_diff = 0.1 * np.cos(2 * np.pi * (t + 0.4)) * 2 * np.pi
    # data_r_diff = 0.1 * np.cos(2 * np.pi * (t + 0.1)) * 2 * np.pi

    def f_prime(x):
        return 0.01 * (0.2 * np.pi * np.cos(0.2 * np.pi * t)
                       + 0.4 * np.pi * np.cos(0.4 * np.pi * t)
                       + 0.75 * np.pi * np.cos(0.75 * np.pi * t)
                       + 1 * np.pi * np.cos(1 * np.pi * t)
                       + 1.5 * np.pi * np.cos(1.5 * np.pi * t))

    return f_prime(t), f_prime(t + config.phase_shift)


def read_data():
    """
    Read data from file or generate simulated data. Which data is used is determined by the Config.excitation value.
    :return: tuple of left and right track data and time values
    """

    x_vals = np.arange(0, config.t_end + 2 * config.phase_shift, config.delta_t)

    if config.data_source == config.TrainData.DATA:
        with open(config.data_r_path, "r") as file:
            data_r = file.readlines()

        data_r = [float(x) for x in data_r]
        data_r = np.array(data_r[:len(x_vals)])

        with open(config.data_l_path, "r") as file:
            data_l = file.readlines()

        data_l = [float(x) for x in data_l]
        data_l = np.array(data_l[:len(x_vals)])

    else:
        x_vals = np.arange(0, config.t_end + 2 * config.phase_shift, config.delta_t_simulation)
        data_l, data_r = simulated_excitation(x_vals)

    return data_l, data_r, x_vals


def _plot_data():
    """
    Plot the data which read_data() returns.
    """

    data_l, data_r, x_vals = read_data()

    plt.figure()

    plt.plot(x_vals, data_l, label="Left")
    plt.plot(x_vals, data_r, label="Right")

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Data")


def _plot_freq():
    """
    Plot the frequency spectrum of the data which read_data() returns using FFT.
    :return:
    """

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

    plt.show()
