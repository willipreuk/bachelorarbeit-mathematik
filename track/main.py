import matplotlib.pyplot as plt
import numpy as np

from simulation.track.data import read_data
from simulation.track.neural_network import predict, train_nn

if __name__ == '__main__':
    nn_x = np.arange(0, 10, 0.0001)

    train_nn()

    plt.figure()
    # model data
    plt.plot(nn_x, predict(nn_x), label="Neural network")
    # real data
    data, _, x_vals = read_data()
    plt.plot(x_vals, data, "x", label="Data")

    plt.title('Model vs Data')
    plt.legend()
    plt.show()
    # plt.savefig("plot/data_5.pdf")
