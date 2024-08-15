from typing import Any

import numpy as np
from numpy import ndarray, dtype

file_path = "data/Hhwayli.dat"


def read_data() -> tuple[list[float], ndarray[float, dtype[Any]]]:
    x_vals = np.arange(0, 409.6, 0.05)

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [float(x) for x in data]

    return data, x_vals
