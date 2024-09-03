def f2mon(crot, cforce, force):
    """Conversion of Forces TWO Moments

    :arg crot: position of the centre of rotation
    :arg cforce: position of force marker
    :arg force: force vector
    :return: moment
    """

    rvec = cforce - crot

    return rvec[0] * force[1] - rvec[1] * force[0]