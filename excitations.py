import numpy as np
import scipy as sp
from scipy.optimize import approx_fprime

import config
from neural_network import predict
from simulation.data import read_data, simulated_excitation, simulated_diff_excitation
from simulation.neural_network import load_model

_data_l, _data_r, _x_vals = read_data()
_spline_l = sp.interpolate.CubicSpline(_x_vals, _data_l)
_spline_r = sp.interpolate.CubicSpline(_x_vals, _data_l)

_fine_x_vals = None
_predicted = None
_nn_spline = None

# reinitialize nn on change of these
last_first_dif_weight = None
last_second_dif_wight = None
last_neural_network_predict_delta_t = None

model = None


def time_excitations(t):
    global _predicted, _nn_spline, _fine_x_vals, last_first_dif_weight, last_second_dif_wight, last_neural_network_predict_delta_t, model

    if config.excitation == config.Excitations.DATA_SPLINE:
        return _spline_l(t), _spline_r(t), _spline_l(t, 1), _spline_r(t, 1)

    if config.excitation == config.Excitations.DATA_NEURAL_NETWORK or config.excitation == config.Excitations.SIMULATED_NEURAL_NETWORK:
        if (config.first_diff_weigth, config.second_diff_weigth, config.neural_network_predict_delta_t) != (
                last_first_dif_weight, last_second_dif_wight, last_neural_network_predict_delta_t):
            _fine_x_vals = np.arange(0, config.t_end, config.neural_network_predict_delta_t)
            _predicted = predict(_fine_x_vals).flatten()
            _nn_spline = sp.interpolate.CubicSpline(_fine_x_vals, _predicted)
            last_first_dif_weight = config.first_diff_weigth
            last_second_dif_wight = config.second_diff_weigth
            last_neural_network_predict_delta_t = config.neural_network_predict_delta_t

        return _nn_spline(t), _nn_spline(t), _nn_spline(t, 1), _nn_spline(t, 1)

    if config.excitation == config.Excitations.SIMULATED:
        left, right = simulated_excitation(t)

        left_diff, right_diff = simulated_diff_excitation(t)

        return left, left, left_diff, right_diff

    if config.excitation == config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT:
        if (model is None) or (config.first_diff_weigth, config.second_diff_weigth) != (
        last_first_dif_weight, last_second_dif_wight):
            model = load_model()
            last_first_dif_weight = config.first_diff_weigth
            last_second_dif_wight = config.second_diff_weigth

        x_left = np.array([t])
        x_right = x_left + config.phase_shift

        left = model.predict(x_left, verbose=0)[0][0]
        right = model.predict(x_right, verbose=0)[0][0]
        left_dif = approx_fprime(x_left, lambda x: model.predict(x, verbose=0)[0], epsilon=1e-6)[0]
        right_dif = approx_fprime(x_right, lambda x: model.predict(x, verbose=0)[0], epsilon=1e-6)[0]

        return left, right, left_dif, right_dif
