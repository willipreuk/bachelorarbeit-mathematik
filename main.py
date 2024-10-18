import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import config
from simulation.track.data import read_data
from simulation.track.neural_network import predict, train_nn


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
    # real data
    data, _, x_vals = read_data()

    # config.error_weight = 1
    # train_nn()
    config.error_weight = 0.99
    train_nn()
    # config.error_weight = 0.9
    # train_nn()
    # config.error_weight = 0.75
    # train_nn()
    # config.error_weight = 0.5
    # train_nn()
    # config.error_weight = 0.25
    # train_nn()
    # config.error_weight = 0.1
    # train_nn()

    plt.figure()

    plot_x = np.arange(0, 10, config.delta_t_simulation / 10)

    config.error_weight = 1
    plt.plot(plot_x, CubicSpline(nn_x, predict(nn_x))(plot_x), label="Neural network" + f" (error weight: {config.error_weight}/{config.first_diff_weight})")
    config.error_weight = 0.999
    plt.plot(plot_x, CubicSpline(nn_x, predict(nn_x))(plot_x), label="Neural network" + f" (error weight: {config.error_weight}/{config.first_diff_weight})")
    config.error_weight = 0.99
    plt.plot(plot_x, CubicSpline(nn_x, predict(nn_x))(plot_x), label="Neural network" + f" (error weight: {config.error_weight}/{config.first_diff_weight})")
    # config.error_weight = 0.75
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.5
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.25
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.1
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")


    plt.plot(x_vals, data, "x", label="Data")
    plt.plot(plot_x, CubicSpline(x_vals, filter_data(data, 10, d=x_vals[1]-x_vals[0]))(plot_x), label="FFT")

    plt.xlabel("Time")
    plt.title('Model vs Data')
    plt.legend()
    plt.show()
    # plt.savefig("plot/comparison-4.pdf")
