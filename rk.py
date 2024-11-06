import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def rk2_constant_step(f, y0, t0, t_end, step_size):
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = np.array(y0, dtype=float)
    h = step_size

    c = np.array([0, 1 / 2])
    a = np.array([
        [0, 0],
        [1 / 2, 0],
    ])
    b = np.array([0, 1])

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        k1 = f(t + c[0] * h, y)
        k2 = f(t + c[1] * h, y + h * (a[1, 0] * k1))

        y = y + h * (b[0] * k1 + b[1] * k2)

        t += h
        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)


def rk4_constant_step(f, y0, t0, t_end, step_size):
    c = np.array([0, 1 / 2, 1 / 2, 1])
    a = np.array([
        [0, 0, 0, 0],
        [1 / 2, 0, 0, 0],
        [0, 1 / 2, 0, 0],
        [0, 0, 1, 0]
    ])
    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])

    t_values = [t0]
    y_values = [y0]

    t = t0
    y = np.array(y0, dtype=float)
    h = step_size

    while t < t_end:
        if t + h > t_end:
            h = t_end - t

        k1 = h * f(t + c[0] * h, y)
        k2 = h * f(t + c[1] * h, y + a[1][0] * k1)
        k3 = h * f(t + c[2] * h, y + a[2][0] * k1 + a[2][1] * k2)
        k4 = h * f(t + c[3] * h, y + a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3)

        y = y + b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4

        t += h
        t_values.append(t)
        y_values.append(y)

    return t_values, y_values


if __name__ == '__main__':
    G = 100
    k = 0.007


    def f(t, y):
        return k * y * (G - y)


    y0 = 1
    t0 = 0
    t_end = 20
    step_size = 1

    t_values, y_values = rk4_constant_step(f, y0, t0, t_end, step_size)
    ref_sol = sp.integrate.solve_ivp(f, (t0, t_end), [y0], method='RK45', t_eval=t_values)

    print(y_values[5])
    print(ref_sol.y[0][5])

    print("Error: ", np.mean(np.abs(y_values - ref_sol.y[0])))

    plt.plot(t_values, y_values, label='solution')
    plt.plot(ref_sol.t, ref_sol.y[0], "--", label='reference solution')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()
