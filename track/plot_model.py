import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from simulation.track.config import Config, Excitations
from simulation.track.eom import eval_eom_ini, eval_eom_ode
from simulation.track.excitations import time_excitations
from simulation.track.model_params import Params


def plot():
    t = np.linspace(0, Config.t_end, 1000)
    q_ini = eval_eom_ini()
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini
    print(q_ini)

    rk45 = sp.integrate.RK45(eval_eom_ode, 0, q_0, Config.t_end, atol=1e-8, rtol=1e-8)

    t_eval = []
    step_sizes = []
    y = []
    while rk45.status == "running":
        rk45.step()
        step_sizes.append(rk45.step_size)
        t_eval.append(rk45.t)
        y.append(rk45.y)

    y = np.transpose(y)

    plt.figure()
    plt.title("step sizes")
    plt.plot(t_eval, step_sizes, label="h")

    plt.plot(t_eval, time_excitations(t_eval)[0], label="left")
    plt.plot(t_eval, time_excitations(t_eval)[1], label="right")
    plt.legend()

    plt.figure()
    plt.title("States")
    plt.plot(t_eval, y[0], label="z_a")
    plt.plot(t_eval, y[1], label="z_s")
    plt.legend()

if __name__ == '__main__':
    plot()
    plt.show()
