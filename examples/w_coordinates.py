"""Cartesian Collins–Soper coordinates for pre-QED W decay four-vectors.

Vectors are (px, py, pz, E), in GeV. This uses the massless convention of the
booked W measurement functions, with the z and y axes signed by boson pz.
The charged lepton defines the angle for both charges. At zero recoil the
azimuthal axis is undefined; rho_sx and rho_sy have a continuous zero limit.
"""
import numpy as np


def coordinates(lepton, neutrino):
    lepton, neutrino = np.asarray(lepton, float), np.asarray(neutrino, float)
    if (lepton.ndim != 2 or lepton.shape[1] != 4 or neutrino.shape != lepton.shape
            or not np.isfinite(lepton).all() or not np.isfinite(neutrino).all()):
        raise ValueError("finite lepton/neutrino arrays must have shape (events, 4)")
    pair = lepton + neutrino
    mass2 = pair[:, 3]**2 - np.sum(pair[:, :3]**2, axis=1)
    if (mass2 <= 0).any():
        raise ValueError("the lepton-neutrino system must have positive invariant mass")
    mass = np.sqrt(mass2)
    qt = np.hypot(pair[:, 0], pair[:, 1])
    transverse_mass = np.hypot(mass, qt)
    sign = np.where(pair[:, 2] < 0, -1., 1.)
    c = sign * ((lepton[:, 3] + lepton[:, 2]) * (neutrino[:, 3] - neutrino[:, 2])
                - (lepton[:, 3] - lepton[:, 2]) * (neutrino[:, 3] + neutrino[:, 2])) / (mass * transverse_mass)
    c = np.clip(c, -1, 1)
    dot = lepton[:, 0] * pair[:, 0] + lepton[:, 1] * pair[:, 1]
    cross = (-lepton[:, 0] * pair[:, 1] + lepton[:, 1] * pair[:, 0]) * sign
    ez = (lepton[:, 3] * pair[:, 3] - lepton[:, 2] * pair[:, 2]) / transverse_mass
    rho = qt / transverse_mass
    rho_sx = 2 * (transverse_mass * dot - qt**2 * ez) / (mass2 * transverse_mass)
    rho_sy = 2 * cross / (mass * transverse_mass)
    sx = np.divide(rho_sx, rho, out=np.zeros_like(rho), where=qt > 1e-12)
    sy = np.divide(rho_sy, rho, out=np.zeros_like(rho), where=qt > 1e-12)
    pt_l = np.hypot(lepton[:, 0], lepton[:, 1])
    pt_nu = np.hypot(neutrino[:, 0], neutrino[:, 1])
    mt = np.sqrt(np.maximum(2 * (pt_l * pt_nu - lepton[:, 0] * neutrino[:, 0]
                                 - lepton[:, 1] * neutrino[:, 1]), 0))
    y = np.arctanh(pair[:, 2] / pair[:, 3])
    return dict(m_w=mass, pt_w=qt, y_w=y, y_abs=np.abs(y), c=c, ctw1=c + 1,
                sx=sx, sy=sy, rho=rho, rho_sx=rho_sx, rho_sy=rho_sy,
                pt_l=pt_l, pt_nu=pt_nu, mt=mt,
                y_l=np.arctanh(lepton[:, 2] / lepton[:, 3]),
                eta_l=np.arcsinh(lepton[:, 2] / pt_l))
