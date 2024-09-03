import numpy as np
from params import UPar, Params


def time_excitations(t):
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
