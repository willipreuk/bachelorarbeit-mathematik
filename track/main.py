import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import alpha

import config
from simulation.track.data import read_data
from simulation.track.neural_network import predict, train_nn


if __name__ == '__main__':
    nn_x = np.arange(0, 10, config.delta_t_simulation / 10)

    config.error_weight = 0.999
    train_nn()
    # config.error_weight = 0.95
    # train_nn()
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
    config.error_weight = 1
    plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    config.error_weight = 0.999
    plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.9
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.75
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.5
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.25
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")
    # config.error_weight = 0.1
    # plt.plot(nn_x, predict(nn_x), label="Neural network" + f" (error weight: {config.error_weight})")

    # real data
    data, _, x_vals = read_data()
    plt.plot(x_vals, data, "x", label="Data", alpha=0.25)

    plt.title('Model vs Data')
    plt.legend()
    plt.show()
    # plt.savefig("plot/comparison-4.pdf")
