import numpy as np
import scipy as sp

from config import Config, Excitations
from neural_network import predict
from simulation.track.data import read_data, simulated_excitation, simulated_diff_excitation

_data_l, _data_r, _x_vals = read_data()
_spline_l = sp.interpolate.CubicSpline(_x_vals, _data_l)
_spline_r = sp.interpolate.CubicSpline(_x_vals, _data_r)

_fine_x_vals = np.arange(0, Config.t_end, Config.delta_t / 100)
_predicted = predict(_fine_x_vals).flatten()
_nn_spline = sp.interpolate.CubicSpline(_fine_x_vals, _predicted)


def time_excitations(t):
    if Config.excitation == Excitations.DATA_SPLINE:
        return _spline_l(t), _spline_r(t), _spline_l(t, 1), _spline_r(t, 1)

    if Config.excitation == Excitations.DATA_NEURAL_NETWORK:
        return _nn_spline(t), _nn_spline(t), _nn_spline(t, 1), _nn_spline(t, 1)

    if Config.excitation == Excitations.SIMULATED:
        left, right = simulated_excitation(t)
        left_diff, right_diff = simulated_diff_excitation(t)

        return left, right, left_diff, right_diff
