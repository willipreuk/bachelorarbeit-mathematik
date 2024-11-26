import matplotlib.pyplot as plt
import numpy as np

import config
from simulation.data import read_data
from simulation.neural_network import train_nn
from plot_model import set_size

def filter_data(data, cutoff_freq, d):
    # Perform FFT
    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), d=d)

    # Apply filter (e.g., low-pass filter)
    fft_result[np.abs(fft_freq) > cutoff_freq] = 0

    # Perform inverse FFT
    filtered_data = np.fft.ifft(fft_result)
    return filtered_data


if __name__ == '__main__':
    nn_x = np.arange(0, 10, config.delta_t_simulation)
    data, _, x_vals = read_data()

    plt.figure(figsize=set_size())

    config.second_diff_weight = 0
    config.first_diff_weight = 0
    history = train_nn()
    plt.loglog(history.history['loss'], label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.01
    history = train_nn()
    plt.loglog(history.history['loss'], label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.1
    history = train_nn()
    plt.loglog(history.history['loss'], label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.25
    history = train_nn()
    plt.loglog(history.history['loss'], label=fr'$\alpha={config.first_diff_weight}$')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plot/plot_train_alpha.pgf', format='pgf')

    plot_x = np.arange(0, 10, config.delta_t_simulation / 10)

    # config.first_diff_weigth = 0
    # config.first_diff_weigth = 0
    # plt.plot(plot_x, CubicSpline(nn_x, predict(nn_x))(plot_x), label=r"$\alpha=0$")
    # config.first_diff_weigth = 0.01
    # plt.plot(plot_x, CubicSpline(nn_x, predict(nn_x))(plot_x), label=r"$\alpha=0.01$")
    # config.second_diff_weigth = 0.1
    # plt.plot(plot_x, CubicSpline(nn_x, predict(nn_x))(plot_x), label=r"$\alpha=0.1$")
    # config.second_diff_weigth = 0.25
    # plt.plot(plot_x, CubicSpline(nn_x, predict(nn_x))(plot_x), label=r"$\alpha=0.25$")
    # config.error_weight = 0.5
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.25
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.1
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")


    # plt.plot(x_vals, data, "x", label="Data", alpha=0.5)
    # plt.plot(plot_x, CubicSpline(x_vals, filter_data(data, 10, d=x_vals[1]-x_vals[0]))(plot_x), label="FFT")

    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.show()
