import numpy as np

def rot_vector(crot, crotp, phi, phip, vec0):
    """Planar Rotation and shift of Vectors.

    :arg crot: position of the centre of rotation
    :arg crotp: velocity of the centre of rotation
    :arg phi: angle of rotation
    :arg phip: time derivative of the angle of rotation
    :arg vec0: input vector

    :returns: position vector in Cartesian coordinates
    :returns: velocity vector in Cartesian coordinates
    """

    a_rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    a_rot_p = phip * np.array([[-np.sin(phi), -np.cos(phi)], [np.cos(phi), -np.sin(phi)]])

    vec = crot + a_rot @ vec0
    vecp = crotp + a_rot_p @ vec0

    return vec, vecp