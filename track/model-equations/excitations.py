import numpy as np
import scipy as sp
from params import UPar, Params


class Excitations:
    DATA_LEFT_PATH = "../data/Hhwayli.dat"
    DATA_RIGHT_PATH = "../data/Hhwayre.dat"

    def __init__(self):
        self.data_time_vals = np.arange(0, 737.221, 0.089993)
        self.data_x_vals = np.arange(0, 409.6, 0.05)

        with open("../data/Hhwayli.dat", "r") as file:
            lines = file.readlines()
            self.data_l = [float(x) for x in lines]

        with open("../data/Hhwayre.dat", "r") as file:
            lines = file.readlines()
            self.data_r = [float(x) for x in lines]

        self.spline_l = sp.interpolate.CubicSpline(self.data_time_vals, self.data_l)
        self.spline_r = sp.interpolate.CubicSpline(self.data_time_vals, self.data_r)

    def spline_time_excitations(self, t):
        """Evaluation of time excitations as measured and interpolated with a cubic spline.

        :param t:
        :return:
        """

        return self.spline_l(t), self.spline_r(t), self.spline_l(t, 1), self.spline_r(t, 1)


    def sim_time_excitations(self, t):
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


if __name__ == '__main__':
    print(sim_time_excitations(0))
