import numpy as np


def gamma(t):
    return [t, t ** 2]


def gamma_dot(t):
    return np.sqrt(1 + 2 * t ** 2)


def calculate_z(x, y):
    # return x * (1 - y) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2 # A
    # return np.cosh(y + np.pi) * np.sin(x + np.pi) + x * (1 - y) # B
    return 10 * (np.cos(x) ** 2 + np.sin(np.pi + y ** 2) ** 2) + np.tanh(np.pi + x ** 2) ** 2 # C


def weg(t):
    x, y = gamma(t)
    return calculate_z(x, y) * gamma_dot(t)
