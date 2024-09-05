import matplotlib.pyplot as plt
import numpy as np

from data import read_data, plot_freq
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

    nn_x = np.arange(0, 409.6, 0.05)
    predicted = nn.predict(nn_x)
    plot_freq(predicted, nn_x[1] - nn_x[0])

    plt.figure()
    plt.plot(nn_x, nn.predict(nn_x))
    plt.plot(x_vals, data, 'x')
    plt.show()
