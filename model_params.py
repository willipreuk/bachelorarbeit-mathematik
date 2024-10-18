import numpy as np


class Params:
    # general data
    nq        = 4               #           number of position coordinates

    g         = 9.81            # [m/s^2]   gravitational constant
    v         = 2.0             # [m/s]     velocity
    te        = 10.0            # [s]       end time

    q_0       = np.array([1.321364, 1.193035, 0.035618, 0.035388])

    # geometrical data
    hha       = 2.8             # [m]       overall height of car
    hsa       = 0.3             # [m]       vertical displacement couple markers car
    hza       = 1.3             # [m]       vertical displacement of wheels
    hfa       = 0.0             # [m]       end point vertical spring at the car (to be determined)
    hss       = 0.105           # [m]       vertical displacement couple markers device
    hfs       = 0.27            # [m]       end point vertical spring at the device

    sshalb    = 9.0             # [m]       halfwidth of device
    shalb     = 0.9             # [m]       horizontal displacement of wheels
    bahalb    = 0.6             # [m]       horizontal displacement couple markers car
    bshalb    = 0.2             # [m]       horizontal displacement couple markers device

    # mass and inertia data
    ms        = 570.0           # [kg]      mass car
    Js        = 6000.0          # [kg*m^2]  moment of inertia car
    ma        = 5500.0          # [kg]      mass device
    Ja        = 500.0           # [kg*m^2]  moment of inertia device

    # parameters of force elements
    cr        = 1.0e6           # [N/m]     spring constant of tyres
    dr        = 0.05            # [-]       damping rate of tyres
    clf       = 87000.0         # [N/m]     spring constant of vertical spring
    dlf       = 0.45            # [-]       damping rate of vertical spring
    cqf       = 6300.0          # [N/m]     spring constant of horizontal springs
    dqf       = 0.45            # [-]       damping rate of horizontal springs


class UPar:
    # parameters of external excitations
    type       = 'harmonic'      #           [ 'harmonic' 'jump' 'user data' ]
    ampl       = 0.1             # [m]       amplitude
    wavelen    = 2.0             # [m]       wavelength
    phas_l     = 0.4             # [-]       phase shift harmonic excitation left wheel
    phas_r     = 0.1             # [-]       phase shift harmonic excitation left wheel
    jumppos    = 1.0 * Params.v  # [m]       position of jump event