import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from params import Params
from eom import eval_eom_ode, eval_eom_ini


def plot():
    t = np.linspace(0, Params.te, 1000)
    q_ini = eval_eom_ini()
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini
    print(q_ini)

    sol = sp.integrate.solve_ivp(eval_eom_ode, [0, Params.te], q_0, t_eval=t, rtol=1e-8, atol=1e-8)


    plt.plot(t, sol.y[0])
    plt.plot(t, sol.y[1])
    plt.plot(t, np.rad2deg(sol.y[2]))
    plt.plot(t, np.rad2deg(sol.y[3]))
    plt.show()


if __name__ == '__main__':
    plot()