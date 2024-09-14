import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from simulation.track.config import Config, Excitations
from simulation.track.eom import eval_eom_ini, eval_eom_ode
from simulation.track.model_params import Params


def plot():
    t = np.linspace(0, Config.t_end, 1000)
    q_ini = eval_eom_ini()
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini
    print(q_ini)

    sol = sp.integrate.solve_ivp(eval_eom_ode, [0, Params.te], q_0, t_eval=t)

    plt.figure()
    plt.plot(t, sol.y[0], label="z_a")
    plt.plot(t, sol.y[1], label="z_s")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
