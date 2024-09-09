import numpy as np

from excitations import Excitations
from rot_vector import rot_vector
from force_spring import f_spring
from force_moments import f2mon
from params import Params

excitation = Excitations()
time_excitation = excitation.sim_time_excitations

def eval_eom(t, q, qp, param_hfa=Params.hfa):
    """Evaluation of equations of motion.

    :arg t: time
    :arg q: position vector [ z_a; z_s; \phi_a; \phi_s ]
    :arg qp: velocity vector [ \dot{z}_a; \dot{z}_s; \dot{\phi}_a; \dot{\phi}_s ]
    :param param_hfa:
    """

    z_a = q[0]
    z_s = q[1]
    phi_a = q[2]
    phi_s = q[3]
    zp_a = qp[0]
    zp_s = qp[1]
    phip_a = qp[2]
    phip_s = qp[3]

    # evaluate external excitations
    ur_l, ur_r, urp_l, urp_r = time_excitation(t)

    # compute all couple markers and their time derivatives
    crot_a = np.array([0, z_a]).transpose()
    crotp_a = np.array([0, zp_a]).transpose()
    crot_s = np.array([0, z_s + Params.hfs]).transpose()
    crotp_s = np.array([0, zp_s]).transpose()

    qr_l, qrp_l = rot_vector(crot_a, crotp_a, phi_a, phip_a, np.array([Params.shalb, - Params.hza]).transpose())
    qr_r, qrp_r = rot_vector(crot_a, crotp_a, phi_a, phip_a, np.array([-Params.shalb, - Params.hza]).transpose())

    qsa_l, qsap_l = rot_vector(crot_a, crotp_a, phi_a, phip_a, np.array([Params.bahalb, - Params.hsa]).transpose())
    qsa_r, qsap_r = rot_vector(crot_a, crotp_a, phi_a, phip_a, np.array([-Params.bahalb, - Params.hsa]).transpose())

    qss_l, qssp_l = rot_vector(crot_s, crotp_s, phi_s, phip_s,
                               np.array([Params.bshalb, -Params.hfs - Params.hss]).transpose())
    qss_r, qssp_r = rot_vector(crot_s, crotp_s, phi_s, phip_s,
                               np.array([Params.bshalb, - Params.hfs - Params.hss]).transpose())

    qva, qvap = rot_vector(crot_a, crotp_a, phi_a, phip_a, np.array([0, -param_hfa]).transpose())
    qvs, qvsp = rot_vector(crot_s, crotp_s, phi_s, phip_s, np.array([0, 0]).transpose())

    # contact points wheel in the inertial system (set to be exactly below wheel)
    qrini_l = np.array([qr_l[0], ur_l]).transpose()
    qrinip_l = np.array([qrp_l[0], urp_l]).transpose()
    qrini_r = np.array([qr_r[0], ur_r]).transpose()
    qrinip_r = np.array([qrp_r[0], urp_r]).transpose()

    # spring forces and momenta
    fr_l = f_spring(Params.cr, Params.dr, Params.ma, qr_l, qrp_l, qrini_l, qrinip_l)
    fr_r = f_spring(Params.cr, Params.dr, Params.ma, qr_r, qrp_r, qrini_r, qrinip_r)

    fs_l = f_spring(Params.cqf, Params.dqf, (Params.ms + Params.ma), qss_l, qssp_l, qsa_l, qsap_l)
    fs_r = f_spring(Params.cqf, Params.dqf, (Params.ms + Params.ma), qss_r, qssp_r, qsa_r, qsap_r)

    c_from = np.array([qva[0], z_s + np.cos(phi_s) * Params.hfs]).transpose()
    c_fromp = np.array([qvap[0], zp_s - np.sin(phi_s) * phip_s * Params.hfs]).transpose()

    fv = f_spring(Params.clf, Params.dlf, Params.ms, c_from, c_fromp, qva, qvap)
    fv[0] = 0

    mr_l = f2mon(crot_a, qr_l, fr_l)
    mr_r = f2mon(crot_a, qr_r, fr_r)

    msa_l = f2mon(crot_a, qsa_l, -fs_l)
    msa_r = f2mon(crot_a, qsa_r, -fs_r)
    mss_l = f2mon(crot_s, qss_l, fs_l)
    mss_r = f2mon(crot_s, qss_r, fs_r)

    mva = f2mon(crot_a, qva, -fv)
    mvs = f2mon(crot_s, qvs, fv)

    zpp_a = (fr_l[1] + fr_r[1] - fs_l[1] - fs_r[1] - fv[1]) / Params.ma
    zpp_s = (fs_l[1] + fs_r[1] + fv[1]) / Params.ms - Params.g

    phipp_a = (mr_l + mr_r + msa_l + msa_r + mva) / Params.Ja
    phipp_s = (mss_l + mss_r + mvs) / (Params.Js + Params.ms * Params.hfs ** 2)

    return np.array([zpp_a, zpp_s, phipp_a, phipp_s]).transpose()


def eval_eom_ode(t, xx):
    """Evaluation of equations of motion for ODE solver.

    :arg t: time
    :arg xx: actual state vector
    :returns: derivative of state vector
    """

    q = xx[0:4]
    qp = xx[4:8]
    qpp = eval_eom(t, q, qp)
    xxp = np.zeros(2 * Params.nq)

    xxp[0:4] = qp
    xxp[4:8] = qpp

    return xxp


def eval_eom_ini():
    """Evaluation of initial conditions for the equations of motion.

    :returns: vector of initial position coordinates [ z_a; z_s; \phi_a; \phi_s ]
    """

    time = 0

    q_act = np.array([Params.hza, Params.hza - Params.hsa + Params.hss, 0, 0])
    qp_act = np.zeros(Params.nq)

    tol = 1e-10
    max_iterations = 20
    max_norm = 1.0e8
    eps = np.finfo(float).eps

    iteration = 0
    act_norm = 0.0

    param_hfa = Params.hfa

    while act_norm > tol * (1 + np.linalg.norm(q_act)) or iteration == 0:
        qactpp0 = eval_eom(time, q_act, qp_act, param_hfa)

        jacobian = np.zeros([Params.nq, Params.nq])

        for j in range(Params.nq):
            if j == 1:
                delta = np.sqrt(eps) * np.max([np.abs(param_hfa), np.sqrt(np.sqrt(eps))])
                q_save = param_hfa
                param_hfa = param_hfa + delta
            else:
                delta = np.sqrt(eps) * np.max([np.abs(q_act[j]), np.sqrt(np.sqrt(eps))])
                q_save = q_act[j]
                q_act[j] = q_act[j] + delta

            qactpp = eval_eom(time, q_act, qp_act, param_hfa)
            jacobian[:, j] = (qactpp - qactpp0) / delta

            if j == 1:
                param_hfa = q_save
            else:
                q_act[j] = q_save

        q_new = np.linalg.solve(jacobian, qactpp0)
        act_norm = np.linalg.norm(q_new)

        if act_norm > max_norm:
            print("Newton iteration did not converge")
            return None

        q_act = q_act - q_new
        q_act[1] = q_act[0] - Params.hsa + Params.hss
        param_hfa = param_hfa - q_new[1]

        iteration += 1

        if iteration > max_iterations:
            print("Newton iteration did not converge")
            return None

    q = q_act
    q[1] = q[0] - Params.hsa + Params.hss

    return q
