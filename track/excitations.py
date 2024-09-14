import numpy as np
import scipy as sp

from config import Config, Excitations
from neural_network import predict
from simulation.track.data import read_data, simulated_excitation, simulated_diff_excitation

data_l, data_r, x_vals = read_data()
spline_l = sp.interpolate.CubicSpline(x_vals, data_l)
spline_r = sp.interpolate.CubicSpline(x_vals, data_r)

fine_x_vals = np.arange(0, Config.t_end, Config.delta_t / 100)
predicted = predict(fine_x_vals).flatten()
spline = sp.interpolate.CubicSpline(fine_x_vals, predicted)


def time_excitations(t):
    if Config.excitation == Excitations.DATA_SPLINE:
        return spline_l(t), spline_r(t), spline_l(t, 1), spline_r(t, 1)

    if Config.excitation == Excitations.DATA_NEURAL_NETWORK:
        return spline(t), spline(t), spline(t, 1), spline(t, 1)

    if Config.excitation == Excitations.SIMULATED:
        left, right = simulated_excitation(t)
        left_diff, right_diff = simulated_diff_excitation(t)

        return left, right, left_diff, right_diff
