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
        integral += (7 * y[i] + 32 * y[i + 1] + 12 * y[i + 2] + 32 * y[i + 3] + 7 * y[i + 4])

    integral *= (4 * h / 90)

    return integral


def gauss_legendre_integration(f, a, b):
    nodes = np.array([-1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7)),
                      -1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                      0.0,
                      1 / 3 * np.sqrt(5 - 2 * np.sqrt(10 / 7)),
                      1 / 3 * np.sqrt(5 + 2 * np.sqrt(10 / 7))
                      ])
    weights = np.array([(322-13*np.sqrt(70))/900,
                        (322+13*np.sqrt(17))/900,
                        128/225,
                        (322+13*np.sqrt(17))/900,
                        (322-13*np.sqrt(70))/900
                        ])

    # Transform interval from [a, b] to [-1, 1]
    def transform(x):
        return 0.5 * (b - a) * x + 0.5 * (b + a)

    integral = 0.0
    for i in range(len(nodes)):
        integral += weights[i] * f(transform(nodes[i]))

    # Scale the result by the interval length
    integral *= 0.5 * (b - a)

    return integral


def gauss_legendre_integration_multiple_intervals(f, a, b, num_intervals):
    # Calculate the width of each subinterval
    h = (b - a) / num_intervals
    total_integral = 0.0

    for i in range(num_intervals):
        sub_a = a + i * h
        sub_b = sub_a + h
        total_integral += gauss_legendre_integration(f, sub_a, sub_b)

    return total_integral


def function(x):
    return 1 / (x + 0.00001)


iterations = 10000

correct = sp.integrate.quad(function, 0, 1)[0]
print("Correct value: ", correct)
print("Error Trapez: ", correct - trapez_sum(0, 1, iterations, function))
print("Error Milne: ", correct - milne_sum(0, 1, iterations, function))
print("Error Gauß-Christoffel: ", correct - gauss_legendre_integration_multiple_intervals(function, 0, 1, iterations))
