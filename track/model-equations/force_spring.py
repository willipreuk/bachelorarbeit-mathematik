import numpy as np

def f_spring(c, d, m, cfrom, cfromp, cto, ctop):
    """Force of a linear spring.

    :arg c: spring constant
    :arg d: damping rate
    :arg m: mass
    :arg cfrom: position of from marker
    :arg cfromp: velocity of from marker
    :arg cto: position of to marker
    :arg ctop: velocity of to marker
    :returns: force vector
    """

    k = 2 * d * np.sqrt(m * c)
    slen = np.linalg.norm(cfrom - cto)

    f = c * slen + k * np.sum((cto-cfrom) * (ctop-cfromp)) / slen

    return f * (cto - cfrom) / slen