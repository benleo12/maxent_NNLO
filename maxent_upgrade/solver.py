"""Damped Newton minimization of the maximum-entropy convex dual.

This module is independent of processes, observables and file formats.
"""
import numpy as np


def _newton_maxent(Phi, p, mu, l2=1e-4, lam0=None, n_iter=200, tol=1e-10, sigma=None):
    """z-scored Newton on the convex dual; warm-startable. Returns (q, lam, ok).

    sigma (optional, shape [K]): Gaussian SOFT constraints.  The primal is then
        min_q KL(q||p) + 1/2 sum_k (E_q[Phi_k] - mu_k)^2 / sigma_k^2
    whose dual is the hard-constraint dual plus 1/2 sum_k (sigma_k/sd_k)^2 lam_k^2,
    i.e. a per-feature ridge equal to the target variance in z-scored units (the
    global `l2` is added to it).  At the optimum  E_q[Phi_k] - mu_k = -sd_k r_k lam_k
    with r_k = l2 + (sigma_k/sd_k)^2: a moment known to better than the prior's own
    spread of the feature is imposed almost exactly, one known worse than that is
    imposed only in part.  sigma_k = 0 (or None) reproduces the hard solve.
    """
    p = np.asarray(p, float)
    sp = p.sum()
    if abs(sp - 1.0) > 1e-9:
        p = p / sp          # the standardization below needs sum(p)=1
    K = Phi.shape[1]
    m_L = (p[:, None] * Phi).sum(0)
    sd = np.maximum(np.sqrt((p[:, None] * (Phi - m_L)**2).sum(0) + 1e-30), 1e-12)
    Phz = (Phi - m_L) / sd
    muz = (mu - m_L) / sd
    # a constant column (e.g. 'norm') whose target equals its value up to rounding: take the
    # z-target from the column itself, so 1/sd (up to 1e12) cannot amplify that rounding
    c0 = Phi[0]
    fit = (Phi.min(0) == Phi.max(0)) & (np.abs(mu - c0) <= 1e-9 * np.maximum(1.0, np.abs(c0)))
    muz[fit] = Phz[0, fit]
    l2v = np.full(K, float(l2))
    if sigma is not None:
        sg = np.nan_to_num(np.asarray(sigma, float), nan=0.0)
        if sg.shape != (K,):
            raise ValueError(f"sigma must have shape ({K},), got {sg.shape}")
        if (sg < 0).any():
            raise ValueError("sigma must be non-negative")
        l2v = l2v + (sg / sd)**2
    lam = np.zeros(K) if lam0 is None else np.asarray(lam0, float).reshape(-1).copy()
    if lam.size != K:
        raise ValueError(f"lam0 has {lam.size} multipliers but the problem has {K} constraints: "
                         "re-solve with the central solve's fixed_orders")
    for it in range(n_iter):
        sarr = Phz @ lam
        smax = sarr.max()
        Z = (p * np.exp(sarr - smax)).sum()
        q = p * np.exp(sarr - smax) / Z
        grad = q @ Phz - muz + l2v * lam
        g = float(np.linalg.norm(grad))
        if not np.isfinite(g):
            return None, lam, False
        if g < tol:
            break
        if g < 1e-6 and not (q > 0).all():
            break        # converged with underflowed weights (fails below)
        Phc = Phz - (q @ Phz)
        H = (q[:, None] * Phc).T @ Phc + np.diag(l2v)
        try:
            step = np.linalg.solve(H + 1e-12 * np.eye(K), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        f0 = smax + np.log(Z) - muz @ lam + 0.5 * lam @ (l2v * lam)
        if g < 1e-6 and float(grad @ step) <= 1e-13 * max(1.0, abs(f0)):
            # the dual cannot resolve this step's decrease: take the full Newton step if it
            # lowers the gradient, otherwise the solve has reached round-off
            ln = lam - step
            sn = Phz @ ln
            qn = p * np.exp(sn - sn.max())
            qn /= qn.sum()
            gn = float(np.linalg.norm(qn @ Phz - muz + l2v * ln))
            if np.isfinite(gn) and gn < g:
                lam = ln
                continue
            break
        t = 1.0
        okls = False
        for _ in range(50):
            ln = lam - t * step
            sn = Phz @ ln
            smn = sn.max()
            if np.isfinite(smn):
                fn = smn + np.log((p * np.exp(sn - smn)).sum()) - muz @ ln + 0.5 * ln @ (l2v * ln)
                if fn <= f0 - 1e-4 * t * float(grad @ step):
                    okls = True
                    break
            t *= 0.5
        if not okls:
            break
        lam = ln
    sarr = Phz @ lam
    smax = sarr.max()
    q = p * np.exp(sarr - smax)
    q /= q.sum()
    g = float(np.linalg.norm(q @ Phz - muz + l2v * lam))     # at the returned multipliers
    ok = bool(np.isfinite(q).all() and (q > 0).all() and np.isfinite(g) and g < 1e-6)
    return q, lam, ok
