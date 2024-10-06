import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import config
from simulation.track.config import t_end
from simulation.track.eom import eval_eom_ode
from simulation.track.excitations import time_excitations
from simulation.track.model_params import Params


def plot():
    q_ini = Params.q_0
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini

    rk45 = sp.integrate.RK45(eval_eom_ode, 0, q_0, config.t_end, atol=1e-8, rtol=1e-8)

    t_eval = []
    step_sizes = []
    y = []
    while rk45.status == "running":
        rk45.step()
        step_sizes.append(rk45.step_size)
        t_eval.append(rk45.t)
        y.append(rk45.y)

    # number of rejected steps is not directly available
    print(len(t_eval), rk45.nfev)

    y = np.array(y)
    step_sizes = np.array(step_sizes)
    t_eval = np.array(t_eval)

    y = np.transpose(y)

    plt.figure()
    plt.title("step sizes and excitations (" + config.excitation.value + ")")
    plt.plot(t_eval, step_sizes, label="h")

    plt.plot(t_eval, time_excitations(t_eval)[0], label="left")
    plt.plot(t_eval, time_excitations(t_eval)[1], label="right")
    plt.legend()

    plt.figure()
    plt.title("States (" + config.excitation.value + ")")
    plt.plot(t_eval, y[0], label="z_a")
    plt.plot(t_eval, y[1], label="z_s")
    plt.legend()


if __name__ == '__main__':
    config.excitation = config.Excitations.SIMULATED_SPLINE
    plot()
    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    plot()
    plt.show()
