import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from params import Params
from eom import eval_eom_ode, eval_eom_ini

file_path = "../data/Hhwayli.dat"

def read_data():
    x_vals = np.arange(0, 409.6, 0.05)

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [float(x) for x in data]

    return np.array(data), x_vals


def plot():
    data, x_vals = read_data()

    t = np.linspace(0, Params.te, 1000)
    q_ini = eval_eom_ini()
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini

    print(q_0)
    sol = sp.integrate.solve_ivp(eval_eom_ode, [0, Params.te], q_0, t_eval=t)


    plt.plot(t, sol.y[0])
    plt.show()


if __name__ == '__main__':
    plot()