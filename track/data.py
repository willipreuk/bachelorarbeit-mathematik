from typing import Any
import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt

file_path = "data/Hhwayli.dat"


def read_data() -> tuple[ndarray[float, dtype[Any]], ndarray[float, dtype[Any]]]:
    x_vals = np.arange(0, 409.6, 0.05)

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [float(x) for x in data]

    return np.array(data), x_vals


def read_data_interpolated() -> tuple[ndarray[float, dtype[Any]], ndarray[float, dtype[Any]]]:
    data, x_vals = read_data()
    x_vals_fine = np.arange(0, 409.6, 0.005)

    return np.interp(x_vals_fine, x_vals, data), x_vals_fine

if __name__ == '__main__':
    data, x_vals = read_data()

    plt.figure()
    nn_x = np.linspace(0, 409.6, 10000)
    plt.plot(x_vals, data)
    plt.show()