import numpy as np
import scipy as sp
import matplotlib

matplotlib.use('pgf')
import matplotlib.pyplot as plt
import cProfile, pstats

import config
from eom import eval_eom_ode
from excitations import time_excitations
from model_params import Params
from data import read_data
from neural_network import load_model, train_nn
from rk import rk2_constant_step, rk4_constant_step

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
        r"\usepackage{siunitx}",
    ])
})


def set_size(w_fraction=1.0, h_fraction=1.0) -> tuple[float, float]:
    width_pt = 400
    fig_width_pt = width_pt
    inches_per_pt = 1 / 72

    golden_ratio = (5 ** .5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt * w_fraction
    fig_height_in = fig_width_in * golden_ratio * h_fraction

    return fig_width_in, fig_height_in


def filter_data(data, cutoff_freq, d):
    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), d=d)

    fft_result[np.abs(fft_freq) > cutoff_freq] = 0

    filtered_data = np.fft.ifft(fft_result)
    return filtered_data


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


def get_q0():
    q_ini = Params.q_0_sim if config.data_source == config.TrainData.SIMULATED else Params.q_0
    q_0 = np.zeros(2 * Params.nq)
    q_0[0:4] = q_ini

    return q_0


def discrete_ref_sol(t_eval):
    q_0 = get_q0()

    if t_eval is None:
        _eval = np.linspace(0, config.t_end, 1000)

    config.excitation = config.Excitations.SIMULATED
    solution, step_sizes_t, step_sizes = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12,
                                                        t_eval=t_eval)

    return t_eval, solution.y, step_sizes_t, step_sizes


def continuous_ref_sol():
    q_0 = get_q0()

    config.excitation = config.Excitations.SIMULATED
    solution, _, _ = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12, dense_output=True)

    return solution.sol


def sol():
    q_0 = get_q0()

    t_eval = np.linspace(0, config.t_end, 1000)

    solution, step_sizes_t, step_sizes = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-6, rtol=1e-4,
                                                        t_eval=t_eval)

    return t_eval, solution.y, step_sizes_t, step_sizes


def sol_constant_step_size(step_size, method='RK4'):
    q_0 = get_q0()

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

    plt.plot(step_sizes_t, step_sizes, label="Step size (ref)")
    plt.plot(step_sizes_t, time_excitations(step_sizes_t)[0], "--", label="Excitation (ref)")

    plt.xlabel("Time $t$ [s]")
    plt.ylabel("Step size & Excitation")
    plt.legend()


def plot_comparison_nn_spline_delta_t():
    plt.figure(figsize=set_size())
    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    config.neural_network_predict_delta_t = config.delta_t

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label=r"$\Delta t = 0.025$")

    config.neural_network_predict_delta_t = config.delta_t / 10

    _, _, step_sizes_t, step_sizes = sol()

    plt.plot(step_sizes_t, step_sizes, label=r"$\Delta t = 0.0025$")

    plt.xlabel(r"Zeit $t$ [\unit{s}]")
    plt.ylabel(r"Schrittweite $h$")
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

    plt.xlabel("Time $t$ [s]")
    plt.ylabel("Step size & Excitation")
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

    plt.plot(t_eval, y[0], "--", label="$z_a$ (ref)")
    plt.plot(t_eval, y[1], "--", label="$z_s$ (ref)")

    plt.xlabel("Time $t$ [s]")
    plt.ylabel("Amplitude $z$ [m]")
    plt.legend()


def plot_sol_dif_comparison():
    plt.figure(figsize=set_size())
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

    config.excitation = config.Excitations.SIMULATED
    config.first_diff_weight = 0
    config.second_diff_weight = 0

    t_eval, y, _, _ = sol()

    plt.plot(t_eval, y[0], "--", label="$z_a$ (Referenz)")
    plt.plot(t_eval, y[1], "--", label="$z_s$ (Referenz)")

    plt.xlabel("Zeit $t$ [s]")
    plt.ylabel("Amplitude $z$ [m]")
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

    plt.xlabel("Time $t$ [s]")
    plt.ylabel("Amplitude $z$ [m]")
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
    plt.xlabel("Schrittweite $h$")
    plt.ylabel("Error")
    plt.legend()


def plot_data():
    config.data_source = config.TrainData.DATA

    data_l, data_r, x_vals = read_data()

    plt.figure(figsize=set_size())

    plt.plot(x_vals, data_l, "C1", label="Links", alpha=1)
    plt.plot(x_vals, data_r, "C2", label="Rechts", alpha=1)
    plt.plot(x_vals, data_r, "xC2", alpha=0.3)
    plt.plot(x_vals, data_l, "xC1", alpha=0.3)

    plt.xlabel(r"Zeit $t$ [\unit{s}]")
    plt.ylabel(r"Auslenkung [\unit{m}]")
    plt.legend()


def plot_runtime_rk45():
    plt.figure(figsize=set_size())
    q_0 = get_q0()

    t_eval = np.linspace(0, config.t_end, 1000)

    tol_space = np.logspace(-5.5, -2.0, 10)
    print(tol_space)

    ref_sol = continuous_ref_sol()

    def profile():
        config.first_diff_weight = 0
        config.second_diff_weight = 0

        profiler = cProfile.Profile()
        profiler.enable()
        solution, _, _ = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=tol, rtol=tol, t_eval=t_eval)
        profiler.disable()
        stats = pstats.Stats(profiler)

        return stats.get_stats_profile().total_tt, np.mean(np.abs(ref_sol(t_eval) - solution.y))

    times_predict = []
    error_predict = []
    times_spline = []
    error_spline = []
    for tol in tol_space:
        config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT
        time, error = profile()
        times_predict.append(time)
        error_predict.append(error)

        config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
        time, error = profile()
        times_spline.append(time)
        error_spline.append(error)

    print(error_predict, times_predict)
    print(error_spline, times_spline)
    print(tol_space)
    plt.loglog(error_predict, times_predict, label="NN")
    plt.loglog(error_spline, times_spline, label="NN (spline)")
    plt.xlabel(r"Error")
    plt.ylabel(r"Zeit $t$ [\unit{s}]")
    plt.legend()


def plot_runtime_constant_step():
    plt.figure(figsize=set_size())
    step_sizes = np.logspace(-3, -1.5, 10)
    print(step_sizes)

    ref_sol = continuous_ref_sol()

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT

    def profile(step_size, method="RK4"):
        config.first_diff_weight = 0
        config.second_diff_weight = 0

        profiler = cProfile.Profile()
        profiler.enable()
        t, y = sol_constant_step_size(step_size, method=method)
        profiler.disable()
        stats = pstats.Stats(profiler)

        return stats.get_stats_profile().total_tt, np.mean(np.abs(ref_sol(t) - np.transpose(y)))

    times_rk4 = []
    error_rk4 = []
    times_rk2 = []
    error_rk2 = []
    for h in step_sizes:
        print("Profile step size: ", h)
        time, error = profile(h, method="RK4")
        times_rk4.append(time)
        error_rk4.append(error)

        time, error = profile(h, method="RK2")
        times_rk2.append(time)
        error_rk2.append(error)

    print(step_sizes)
    print(times_rk4)
    print(error_rk4)
    print(times_rk2)
    print(error_rk2)
    plt.loglog(error_rk4, times_rk4, label="RK4")
    plt.loglog(error_rk2, times_rk2, label="RK2")
    plt.xlabel(r"Error")
    plt.ylabel(r"Zeit $t$ [\unit{s}]")
    plt.legend()


def plot_data_fft():
    config.data_source = config.TrainData.DATA
    config.t_end = 10
    data_l, _, x_vals = read_data()

    plt.figure(figsize=set_size())
    y_10 = filter_data(data_l, 10, d=x_vals[1] - x_vals[0])
    y_5 = filter_data(data_l, 5, d=x_vals[1] - x_vals[0])
    y_15 = filter_data(data_l, 15, d=x_vals[1] - x_vals[0])

    plt.plot(x_vals, data_l, "x", label="Messdaten", alpha=0.5)
    plt.plot(x_vals, y_15, label="15Hz")
    plt.plot(x_vals, y_10, label="10Hz")
    plt.plot(x_vals, y_5, label="5Hz")

    plt.xlabel(r"Zeit $t$ [\unit{s}]")
    plt.ylabel(r"Auslenkung [\unit{m}]")
    plt.legend(loc='upper right')


def plot_prediction():
    config.data_source = config.TrainData.DATA
    config.t_end = 10
    data_l, _, x_vals = read_data()
    model = load_model()

    plt.figure(figsize=set_size(h_fraction=0.5))

    plt.plot(x_vals, data_l, "x", label="Messdaten", alpha=0.5)
    plt.plot(x_vals, model.predict(x_vals), label="Vorhersage")

    plt.xlabel(r"Time $t$ [\unit{s}]")
    plt.ylabel(r"Amplitude [\unit{m}]")
    plt.legend()


def plot_model_prediction_alpha():
    config.data_source = config.TrainData.SIMULATED
    config.t_end = 3

    nn_x = np.arange(0, config.t_end, config.delta_t_simulation / 10)
    data, _, x_vals = read_data()
    x_vals = x_vals[x_vals < config.t_end]
    data = data[:len(x_vals)]

    plt.figure(figsize=set_size())

    config.second_diff_weight = 0
    config.first_diff_weight = 0
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.01
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.1
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.25
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\alpha={config.first_diff_weight}$')

    plt.plot(x_vals, data, "x", label="Messdaten", alpha=0.5)

    plt.xlabel(r'Zeit $t$ [\unit{s}]')
    plt.ylabel(r'Auslenkung [\unit{m}]')
    plt.legend()


def plot_model_prediction_alpha_simulated():
    config.data_source = config.TrainData.SIMULATED

    nn_x = np.arange(0, config.t_end, config.delta_t_simulation / 10)
    data, _, x_vals = read_data()
    x_vals = x_vals[x_vals < config.t_end]
    data = data[:len(x_vals)]

    plt.figure(figsize=set_size())

    config.second_diff_weight = 0
    config.first_diff_weight = 0
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.01
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.1
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\alpha={config.first_diff_weight}$')

    plt.plot(x_vals[9::10], data[9::10], "--", label="Messdaten")

    plt.xlabel(r'Zeit $t$ [\unit{s}]')
    plt.ylabel(r'Auslenkung [\unit{m}]')
    plt.legend()


def plot_model_prediction_beta():
    config.data_source = config.TrainData.DATA
    config.t_end = 10
    config.epochs = 3000

    nn_x = np.arange(0, config.t_end, config.delta_t_simulation / 10)
    data, _, x_vals = read_data()
    x_vals = x_vals[x_vals < config.t_end]
    data = data[:len(x_vals)]

    plt.figure(figsize=set_size())

    config.first_diff_weight = 0
    config.second_diff_weight = 0
    train_nn()
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\beta={config.first_diff_weight}$')

    config.second_diff_weight = 0.01
    train_nn()
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\beta={config.first_diff_weight}$')

    config.second_diff_weight = 0.1
    train_nn()
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\beta={config.first_diff_weight}$')

    config.second_diff_weight = 0.25
    train_nn()
    plt.plot(nn_x, load_model().predict(nn_x), label=fr'$\beta={config.first_diff_weight}$')

    plt.plot(x_vals, data, "x", label="Messdaten", alpha=0.5)

    plt.xlabel(r'Zeit $t$ [\unit{s}]')
    plt.ylabel(r'Auslenkung [\unit{m}]')
    plt.legend()


def plot_training_loss():
    plt.figure(figsize=set_size())

    config.data_source = config.TrainData.DATA
    config.t_end = 10
    config.epochs = 3000

    config.second_diff_weight = 0

    config.first_diff_weight = 0
    history = train_nn()
    plt.loglog(history.history["loss"], label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.01
    history = train_nn()
    plt.loglog(history.history["loss"], label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.1
    history = train_nn()
    plt.loglog(history.history["loss"], label=fr'$\alpha={config.first_diff_weight}$')

    config.first_diff_weight = 0.25
    history = train_nn()
    plt.loglog(history.history["loss"], label=fr'$\alpha={config.first_diff_weight}$')

    plt.xlabel("Epochen")
    plt.ylabel("Loss")
    plt.legend()


def plot_sol_err_spline():
    plt.figure(figsize=set_size())

    q_0 = get_q0()

    t_eval = np.linspace(0, config.t_end, 1000)
    h_space = np.linspace(config.delta_t / 10000, config.delta_t, 10)
    print(h_space)

    ref_sol = continuous_ref_sol()

    error_0 = []
    error_1 = []
    error_2 = []
    error_3 = []

    config.excitation = config.Excitations.SIMULATED_NEURAL_NETWORK
    config.second_diff_weight = 0

    for h in h_space:
        print("Step size: ", h)

        config.neural_network_predict_delta_t = h
        config.first_diff_weight = 0
        solution_0, _, _ = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12, t_eval=t_eval)

        config.first_diff_weight = 0.01
        solution_1, _, _ = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12, t_eval=t_eval)

        config.first_diff_weight = 0.1
        solution_2, _, _ = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12, t_eval=t_eval)

        config.first_diff_weight = 0.25
        solution_3, _, _ = custom_sol_ivp(eval_eom_ode, (0, config.t_end), q_0, atol=1e-12, rtol=1e-12, t_eval=t_eval)

        error_0.append(np.mean(np.abs(ref_sol(t_eval) - solution_0.y)))
        error_1.append(np.mean(np.abs(ref_sol(t_eval) - solution_1.y)))
        error_2.append(np.mean(np.abs(ref_sol(t_eval) - solution_2.y)))
        error_3.append(np.mean(np.abs(ref_sol(t_eval) - solution_3.y)))


    print(h_space)
    print(error_0)
    print(error_1)
    print(error_2)
    print(error_3)


    plt.loglog(h_space, error_0, label=r"$\alpha = 0$")
    plt.loglog(h_space, error_1, label=r"$\alpha = 0.01$")
    plt.loglog(h_space, error_2, label=r"$\alpha = 0.1$")
    plt.loglog(h_space, error_3, label=r"$\alpha = 0.25$")

    plt.xlabel(r"Schrittweite $h$")
    plt.ylabel(r"Error")
    plt.legend()


if __name__ == '__main__':
    plot_sol_err_spline()

    plt.tight_layout(pad=0.3)
    plt.savefig('plot/plot_sol_err_spline.pgf', format='pgf')
