import numpy as np


def gamma(t):
    return [t, t ** 2]


def gamma_dot(t):
    return np.sqrt(1 + 2 * t ** 2)


def calculate_z(x, y):
    return np.e ** x * np.cos(y) * np.sin(x + np.pi)


def weg(t):
    x, y = gamma(t)
    return calculate_z(x, y) * gamma_dot(t)
