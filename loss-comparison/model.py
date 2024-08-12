import numpy as np


def gamma(t):
    return [t, t ** 2]


def gamma_dot(t):
    return np.sqrt(1 + 2 * t ** 2)


def calculate_z(x, y):
    return x * (1 - y) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
    # return np.e ** x * np.tanh(y) * np.sin(x + np.pi) + np.cosh(y * 5) * x * (1 - y)


def weg(t):
    x, y = gamma(t)
    return calculate_z(x, y) * gamma_dot(t)
