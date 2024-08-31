import matplotlib.pyplot as plt
import numpy as np

from data import read_data
from neural_network import NeuralNetwork
from simulation.track.data import read_data_interpolated

if __name__ == '__main__':
    data, x_vals = read_data_interpolated()

    nn = NeuralNetwork(x_vals, data)
    # nn.load()

    nn.create_model()
    history = nn.train()

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    nn_x = np.linspace(0, 409.6, 10000)
    plt.plot(nn_x, nn.predict(nn_x))
    plt.plot(x_vals, data, 'x')
    plt.show()
