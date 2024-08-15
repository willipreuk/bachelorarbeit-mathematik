import numpy as np


def gamma(t):
    return [t, t ** 2]


def gamma_dot(t):
    return np.sqrt(1 + 2 * t ** 2)


def get_weg(fun):
    def weg(t):
        x, y = gamma(t)
        return fun(x, y) * gamma_dot(t)

    return weg
