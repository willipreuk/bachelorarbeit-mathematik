import scipy as sp
import numpy as np


def trapez_sum(a, b, n, f):
    h = (b - a) / n

    summe = 0
    for i in range(1, n):
        summe += f(a + i * h)

    return h * (1 / 2 * (f(a) + f(b)) + summe)


def milne_sum(a, b, n, f):
    # Check if n is a multiple of 4
    if n % 4 != 0:
        raise ValueError("Number of intervals (n) must be a multiple of 4")

    h = (b - a) / n

    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Apply Milne's rule
    integral = 0
    for i in range(0, n, 4):
        integral += (7*y[i] + 32*y[i+1] + 12*y[i+2] + 32*y[i+3] + 7*y[i+4])

    integral *= (4 * h / 90)

    return integral


def function(x):
    return 1/(x+0.00001)


iterations = 100000

correct = sp.integrate.quad(function, 0, np.pi)[0]
print("Correct value: ", correct)
print(correct - trapez_sum(0, np.pi, iterations, function))
print(correct - milne_sum(0, np.pi, iterations, function))
