r"""The Chebyshev basis and the variable maps, shared by every part of the package.

    u = umap(x, lo, hi, kind)       # x -> u in [-1, 1]; values outside [lo, hi] clip to the edge
    C = cheb(u, N)                  # C[:, n] = T_n(u), n = 0..N, by T_n = 2u T_{n-1} - T_{n-2}

Maps ('kind'):
  'lin'               u = 2 (x - lo)/(hi - lo) - 1                      (rapidities, angles)
  'log'               u = 2 (ln x - ln lo)/(ln hi - ln lo) - 1          (transverse momenta, continuum masses)
  'bw' or ('bw', M, G) u = 2 (t(x) - t(lo))/(t(hi) - t(lo)) - 1,  t = arctan((x^2 - M^2)/(M G))
                      (resonant masses; 'bw' alone uses the Z mass and width)
The map must be the one compiled into the fixed-order run that produced the moments.
"""
import numpy as np

# Default resonance of the Breit-Wigner map (the Z).  Must match the Fortran constants of
# the NNLOJET patch (eval_chebT_mll / bw_u), or the moments compare different variables.
BW_M, BW_G = 91.1876, 2.4952


def cheb(u, N):
    """Chebyshev polynomials T_0..T_N of u (clipped to [-1, 1]); returns shape (len(u), N+1)."""
    u = np.clip(np.asarray(u, float), -1, 1)
    out = np.zeros((len(u), N + 1)); out[:, 0] = 1.0
    if N >= 1:
        out[:, 1] = u
    for n in range(2, N + 1):
        out[:, n] = 2 * u * out[:, n - 1] - out[:, n - 2]
    return out


def umap(x, lo, hi, kind):
    """Map an observable onto u in [-1, 1] ('lin', 'log', 'bw' or ('bw', M, G)); see module doc."""
    if isinstance(kind, (tuple, list)) and kind and kind[0] == 'bw':
        _, M, G = (list(kind) + [BW_M, BW_G])[:3]
        kind = 'bw'
    else:
        M, G = BW_M, BW_G
    if kind == 'bw':
        f = lambda v: np.arctan((np.asarray(v, float) ** 2 - M * M) / (M * G))
        t, tl, th = f(np.clip(x, lo, hi)), f(lo), f(hi)
        return 2 * (t - tl) / (th - tl) - 1
    if kind == 'log':
        lo = max(lo, 1e-6); xx = np.clip(x, lo, hi)
        return 2 * (np.log(xx) - np.log(lo)) / (np.log(hi) - np.log(lo)) - 1
    if kind != 'lin':
        raise ValueError(f"unknown map '{kind}': use 'lin', 'log', 'bw' or ('bw', M, Gamma)")
    return 2 * (np.clip(x, lo, hi) - lo) / (hi - lo) - 1
