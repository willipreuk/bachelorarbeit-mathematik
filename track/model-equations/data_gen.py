import matplotlib.pyplot as plt
import numpy as np

from params import UPar, Params


def left_excitation(t):
    return UPar.ampl * np.sin(2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_l))


def right_excitation(t):
    return UPar.ampl * np.sin(2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_r))


def left_excitation_derivative(t):
    return 2 * np.pi * (Params.v / UPar.wavelen) * UPar.ampl * np.cos(
        2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_l))


def right_excitation_derivative(t):
    return 2 * np.pi * (Params.v / UPar.wavelen) * UPar.ampl * np.cos(
        2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_r))


def generate_data():
    t = np.arange(0, Params.te, 0.01)
    data = np.array([left_excitation(t), right_excitation(t), left_excitation_derivative(t), right_excitation_derivative(t)]) + np.random.normal(0, 0.01, (4, len(t)))
    return data


if __name__ == '__main__':
    data = generate_data()

    plt.plot(np.arange(0, Params.te, 0.01), data[0], "x")
    plt.plot(np.arange(0, Params.te, 0.01), left_excitation(np.arange(0, Params.te, 0.01)))
    plt.show()