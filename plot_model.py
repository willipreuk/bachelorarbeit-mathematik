import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import config
from simulation.eom import eval_eom_ode
from simulation.excitations import time_excitations
from simulation.model_params import Params
from simulation.data import read_data
from simulation.rk import rk2_constant_step, rk4_constant_step


def custom_sol_ivp(*args, **kwargs):
    """
    Custom solve_ivp function that returns the step sizes used in the simulation.
    :param args: args of solve_ivp
    :param kwargs: kwargs of solve_ivp
    :return:
    """
    step_sizes = []
    step_sizes_t = []

    class Solver(sp.integrate.RK23):
        def __init__(self, *solver_args, **solver_kwargs):
            super().__init__(*solver_args, **solver_kwargs)
            self.step_sizes = []

        def step(self):
            super().step()
            step_sizes.append(self.step_size)
            # print("Step size: ", self.step_size)
            step_sizes_t.append(self.t)

    solution = sp.integrate.solve_ivp(*args, **kwargs, method=Solver)

    return solution, step_sizes_t, step_sizes


def discrete_ref_sol(t_eval):
    q_ini = Params.q_0
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini

    if t_eval is None:
        _eval = np.linspace(0, config.t_end, 1000)

    config.excitation = config.Excitations.SIMULATED
    solution, step_sizes_t, step_sizes = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12, t_eval=t_eval)

    return t_eval, solution.y, step_sizes_t, step_sizes


def continuous_ref_sol():
    q_ini = Params.q_0
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini

    config.excitation = config.Excitations.SIMULATED
    solution, _, _ = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12, dense_output=True)

    return solution.sol



def sol():
    q_ini = Params.q_0
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini

    t_eval = np.linspace(0, config.t_end, 1000)

    solution, step_sizes_t, step_sizes = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-6, rtol=1e-4, t_eval=t_eval)

    return t_eval, solution.y, step_sizes_t, step_sizes


def sol_constant_step_size(step_size, method='RK4'):
    q_ini = Params.q_0
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini

    if method == 'RK4':
        t, y = rk4_constant_step(eval_eom_ode, q_0, 0, config.t_end, step_size)
    elif method == 'RK2':
        t, y = rk2_constant_step(eval_eom_ode, q_0, 0, config.t_end, step_size)
    else:
        raise ValueError("Method not supported")

    return t, y


# These functions are used to plot the figures in the thesis.

def plot_comparison_step_sizes():
    plt.figure(figsize=(10, 5))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label="Step size (NN)")
    plt.plot(step_sizes_t, time_excitations(step_sizes_t)[0], label="Excitation (NN)")

    config.excitation = config.Excitations.SIMULATED
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label="Step size (reference)")
    plt.plot(step_sizes_t, time_excitations(step_sizes_t)[0], "--", label="Excitation (reference)")

    plt.xlabel("Time")
    plt.legend()


def plot_comparison_nn_spline_delta_t():
    plt.figure(figsize=(10, 5))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    config.neural_network_predict_delta_t = config.delta_t

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label=r"$\Delta x = 0.025$")

    config.neural_network_predict_delta_t = config.delta_t / 10

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label=r"$\Delta x = 0.0025$")

    plt.xlabel("Time")
    plt.legend()


def plot_comparison_nn_step_sizes():
    plt.figure(figsize=(10, 5))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label=r"Step size (NN)")
    plt.plot(step_sizes_t, time_excitations(step_sizes_t)[0], label=r"Excitation (NN)")

    config.first_diff_weight = 0.01
    config.second_diff_weight = 0

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label=r"Step size (NN $\alpha = 0.01$)")
    plt.plot(step_sizes_t, time_excitations(step_sizes_t)[0], "--", label=r"Excitation (NN $\alpha = 0.01$)")

    plt.xlabel("Time")
    plt.legend()


def plot_sol_comparison():
    plt.figure(figsize=(10, 5))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], label="$z_a$ (NN)")
    plt.plot(t_eval, y[1], label="$z_s$ (NN)")

    config.excitation = config.Excitations.SIMULATED
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], "--", label="$z_a$ (reference)")
    plt.plot(t_eval, y[1], "--", label="$z_s$ (reference)")

    plt.xlabel("Time")
    plt.legend()


def plot_sol_dif_comparison():
    plt.figure(figsize=(10, 5))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], label=r"$z_a$ (NN $\alpha = 0$)")
    plt.plot(t_eval, y[1], label=r"$z_s$ (NN $\alpha = 0$)")

    config.first_diff_weight = 0.01
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], label=r"$z_a$ (NN $\alpha = 0.01$)")
    plt.plot(t_eval, y[1], label=r"$z_s$ (NN $\alpha = 0.01$)")

    config.first_diff_weight = 0.1
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], label=r"$z_a$ (NN $\alpha = 0.1$)")
    plt.plot(t_eval, y[1], label=r"$z_s$ (NN $\alpha = 0.1$)")

    config.first_diff_weight = 0.25
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], label=r"$z_a$ (NN $\alpha = 0.25$)")
    plt.plot(t_eval, y[1], label=r"$z_s$ (NN $\alpha = 0.25$)")

    config.excitation = config.Excitations.SIMULATED
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], "--", label="$z_a$ (reference)")
    plt.plot(t_eval, y[1], "--", label="$z_s$ (reference)")

    plt.xlabel("Time")
    plt.legend()


def plot_sol_ref_predicted():
    plt.figure(figsize=(10, 5))

    config.excitation = config.Excitations.SIMULATED
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    t_eval, y_ref, _, _ = sol()

    plt.plot(t_eval, y_ref[0], label=r"$z_a$ (ref)")
    plt.plot(t_eval, y_ref[1], label=r"$z_s$ (ref)")

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], "--", label=r"$z_a$ (NN)")
    plt.plot(t_eval, y[1], "--", label=r"$z_s$ (NN)")


    print("Max diff z_a: ", np.max(np.abs(y_ref[0] - y[0])))
    print("Max diff z_s: ", np.max(np.abs(y_ref[1] - y[1])))


    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT
    config.first_diff_weight = 0.01
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], "--", label=r"$z_a$ (NN $\alpha=0.01$)")
    plt.plot(t_eval, y[1], "--", label=r"$z_s$ (NN $alpha=0.01$)")


    print("Max diff z_a: ", np.max(np.abs(y_ref[0] - y[0])))
    print("Max diff z_s: ", np.max(np.abs(y_ref[1] - y[1])))

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT
    config.first_diff_weight = 0.1
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], "--", label=r"$z_a$ (NN $\alpha=0.1$)")
    plt.plot(t_eval, y[1], "--", label=r"$z_s$ (NN $\alpha=0.1$)")


    print("Max diff z_a: ", np.max(np.abs(y_ref[0] - y[0])))
    print("Max diff z_s: ", np.max(np.abs(y_ref[1] - y[1])))

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()


def plot_constant_step_size_error():
    plt.figure()

    config.excitation = config.Excitations.SIMULATED
    config.first_diff_weight = 0.01
    config.second_diff_weight = 0

    ref_solution = continuous_ref_sol()

    step_sizes = np.linspace(0.0001, 0.01, 10)

    errors_rk2 = []
    errors_rk4 = []
    for step_size in step_sizes:
        print("Step size: ", step_size)

        config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT

        t, y = sol_constant_step_size(step_size, method='RK4')
        _, y_rk2 = sol_constant_step_size(step_size, method='RK2')

        y_ref = ref_solution(t)

        errors_rk4.append(np.mean(np.abs(y_ref - np.transpose(y))))
        errors_rk2.append(np.mean(np.abs(y_ref - np.transpose(y_rk2))))


    print("Errors RK2: ", errors_rk2)
    print("Errors RK4: ", errors_rk4)

    plt.loglog(step_sizes, errors_rk2, label="RK2")
    plt.loglog(step_sizes, errors_rk4, label="RK4")
    plt.xlabel("Step size")
    plt.ylabel("Error")
    plt.legend()



def plot_data():
    data_l, data_r, x_vals = read_data()

    plt.figure(figsize=(10, 5))

    plt.plot(x_vals, data_l, "C1", label="Left", alpha=1)
    plt.plot(x_vals, data_r, "C2", label="Right", alpha=1)
    plt.plot(x_vals, data_r, "xC2", alpha=0.3)
    plt.plot(x_vals, data_l, "xC1", alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()


if __name__ == '__main__':
    plot_constant_step_size_error()
    plt.savefig("plot/plot_constant_step_size_error_9.pdf")
