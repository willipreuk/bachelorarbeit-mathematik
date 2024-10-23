import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import config
from simulation.eom import eval_eom_ode
from simulation.excitations import time_excitations
from simulation.model_params import Params
from simulation.data import read_data

def sol():
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

    return t_eval, step_sizes, y


def plot_comparison_step_sizes():
    plt.figure(figsize=(10,5))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.first_diff_weigth = 0
    config.second_diff_weigth = 0

    t_eval, step_sizes, y = sol()

    plt.plot(t_eval, step_sizes, label="Step size (NN)")
    plt.plot(t_eval, time_excitations(t_eval)[0], label="Excitation (NN)")

    config.excitation = config.Excitations.SIMULATED_SPLINE
    config.first_diff_weigth = 0
    config.second_diff_weigth = 0

    t_eval, step_sizes, y = sol()

    plt.plot(t_eval, step_sizes, label="Step size (reference)")
    plt.plot(t_eval, time_excitations(t_eval)[0],"--", label="Excitation (reference)")


    plt.xlabel("Time")
    plt.legend()
    # plt.figure()
    # plt.title("States (" + config.excitation.value + ")")
    # plt.plot(t_eval, y[0], label="z_a")
    # plt.plot(t_eval, y[1], label="z_s")
    # plt.legend()

def plot_sol_comparison():
    plt.figure(figsize=(10,5))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.first_diff_weigth = 0
    config.second_diff_weigth = 0

    t_eval, step_sizes, y = sol()

    plt.plot(t_eval, y[0], label="$z_a$ (NN)")
    plt.plot(t_eval, y[1], label="$z_s$ (NN)")

    config.excitation = config.Excitations.SIMULATED_SPLINE
    config.first_diff_weigth = 0
    config.second_diff_weigth = 0

    t_eval, step_sizes, y = sol()

    plt.plot(t_eval, y[0], "--", label="$z_a$ (reference)")
    plt.plot(t_eval, y[1], "--", label="$z_s$ (reference)")

    plt.xlabel("Time")
    plt.legend()


def plot_data():
    data_l, data_r, x_vals = read_data()

    plt.figure(figsize=(10,5))

    plt.plot(x_vals, data_l, "C1", label="Left", alpha=1)
    plt.plot(x_vals, data_r, "C2", label="Right", alpha=1)
    plt.plot(x_vals, data_r, "xC2", alpha=0.3)
    plt.plot(x_vals, data_l, "xC1", alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()


if __name__ == '__main__':
    plot_comparison_step_sizes()
    plt.savefig("plot/excitation-simulated-h-comparison.pdf")
