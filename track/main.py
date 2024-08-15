import matplotlib.pyplot as plt
import numpy as np

from data import read_data
from neural_network import NeuralNetwork

if __name__ == '__main__':
    data, x_vals = read_data()

    nn = NeuralNetwork(x_vals, data)
    nn.train()

    nn_x = np.linspace(0, 409.6, 10000)
    plt.plot(nn_x, nn.predict(nn_x))
    plt.plot(x_vals, data, 'x')
    plt.show()
