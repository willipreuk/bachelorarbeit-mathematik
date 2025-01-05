import numpy as np
import scipy as sp
from scipy.optimize import approx_fprime

import config
from neural_network import predict
from data import read_data, simulated_excitation, simulated_diff_excitation
from neural_network import load_model


def _filter_data(data, cutoff_freq, sample_rate):
    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), d=1 / sample_rate)

    fft_result[np.abs(fft_freq) > cutoff_freq] = 0

    filtered_data = np.fft.ifft(fft_result)
    return filtered_data


_data_l, _data_r, _x_vals = read_data()
_sample_rate = 1 / (_x_vals[1] - _x_vals[0])
_cutoff_freq = 10.0  # Cutoff frequency of 10 H
_filtered_data_l = _filter_data(_data_l, _cutoff_freq, _sample_rate)
_spline_l = sp.interpolate.CubicSpline(_x_vals, _filtered_data_l)

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
        return _spline_l(t), _spline_l(t + config.phase_shift), _spline_l(t, 1), _spline_l(t + config.phase_shift, 1)

    if config.excitation == config.Excitations.DATA_NEURAL_NETWORK or config.excitation == config.Excitations.SIMULATED_NEURAL_NETWORK:
        if (config.first_diff_weight, config.second_diff_weight, config.neural_network_predict_delta_t) != (
                last_first_dif_weight, last_second_dif_wight, last_neural_network_predict_delta_t):
            _fine_x_vals = np.arange(0, config.t_end + config.phase_shift, config.neural_network_predict_delta_t)
            _predicted = predict(_fine_x_vals).flatten()
            _nn_spline = sp.interpolate.CubicSpline(_fine_x_vals, _predicted)
            last_first_dif_weight = config.first_diff_weight
            last_second_dif_wight = config.second_diff_weight
            last_neural_network_predict_delta_t = config.neural_network_predict_delta_t

        return _nn_spline(t), _nn_spline(t + config.phase_shift), _nn_spline(t, 1), _nn_spline(t + config.phase_shift,
                                                                                               1)

    if config.excitation == config.Excitations.SIMULATED:
        left, right = simulated_excitation(t)

        left_diff, right_diff = simulated_diff_excitation(t)

        return left, right, left_diff, right_diff

    if config.excitation == config.Excitations.SIMULATED_NEURAL_NETWORK_PREDICT or config.excitation == config.Excitations.DATA_NEURAL_NETWORK_PREDICT:
        if (model is None) or (config.first_diff_weight, config.second_diff_weight) != (
                last_first_dif_weight, last_second_dif_wight):
            model = load_model()
            last_first_dif_weight = config.first_diff_weight
            last_second_dif_wight = config.second_diff_weight

        print("t: ", t)

        x_left = np.array([t])
        x_right = x_left + config.phase_shift

        left = model.predict(x_left, verbose=0)[0][0]
        right = model.predict(x_right, verbose=0)[0][0]
        left_dif = approx_fprime(x_left, lambda x: model.predict(x, verbose=0)[0], epsilon=1e-6)[0]
        right_dif = approx_fprime(x_right, lambda x: model.predict(x, verbose=0)[0], epsilon=1e-6)[0]

        return left, right, left_dif, right_dif
