import matplotlib.pyplot as plt
import numpy as np

from simulation.track.neural_network import predict, train_nn

if __name__ == '__main__':
    nn_x = np.arange(0, 10, 0.0001)

    train_nn()

    plt.figure()
    # model data
    plt.plot(nn_x, predict(nn_x), label="NN")

    plt.title('Model vs Data')
    plt.legend()
    plt.show()
