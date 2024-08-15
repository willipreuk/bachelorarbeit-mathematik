import numpy as np
from network import train

functions = {
    'c': lambda x, y: np.sin(20 * x) * np.cos(20 * y + np.pi),
    'd': lambda x, y: np.where(x > 0.5, 1, 0) * np.where(y > 0.5, 1, 0) + np.sin(x),
    'a': lambda x, y: x * (1 - y) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2,
    'b': lambda x, y: np.cosh(y + np.pi) * np.sin(x + np.pi) + x * (1 - y),
}


if __name__ == '__main__':
    for name, fun in functions.items():
        print(f"Function: {name}")

        for i in range(5):
            train(fun, f"results_{name}.csv")