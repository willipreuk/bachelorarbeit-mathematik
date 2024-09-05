import numpy as np
import scipy as sp
from params import UPar, Params


def sim_time_excitations(t):
    """Evaluation of time excitations.

    :arg t: time
    :returns: excitation vector [ ur_l(time); ur_r(time); urp_l(time); urp_r(time) ]
    """

    ur_l = UPar.ampl * np.sin(2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_l))
    ur_r = UPar.ampl * np.sin(2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_r))
    urp_l = 2 * np.pi * (Params.v / UPar.wavelen) * UPar.ampl * np.cos(
        2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_l))
    urp_r = 2 * np.pi * (Params.v / UPar.wavelen) * UPar.ampl * np.cos(
        2 * np.pi * ((Params.v / UPar.wavelen) * t + UPar.phas_r))

    return ur_l, ur_r, urp_l, urp_r


def spline_time_excitations(t):
    """Evaluation of time excitations as measured and interpolated with a cubic spline.

    :param t:
    :return:
    """

    x_vals = np.arange(0, 409.6, 0.05)

    with open("../data/Hhwayli.dat", "r") as file:
        data_l = file.readlines()
        data_l = [float(x) for x in data_l]

    with open("../data/Hhwayre.dat", "r") as file:
        data_r = file.readlines()
        data_r = [float(x) for x in data_r]

    spline_l = sp.interpolate.CubicSpline(x_vals, data_l)
    spline_r = sp.interpolate.CubicSpline(x_vals, data_r)


    return spline_l(t), spline_r(t), spline_l(t, 1), spline_r(t, 1)


if __name__ == '__main__':
    print(sim_time_excitations(0))
    print(spline_time_excitations(0))