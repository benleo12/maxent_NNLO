#!/usr/bin/env python3
r"""Load event-level Chebyshev moments from a Stripper XML (Rene's Z+jet runs)
into the `moments["recoil"][obs]` dict that maxent_upgrade.upgrade() consumes.

File conventions (verified 2026-09-15/16 against the LO test file):
  * every histogram carries under/overflow slots: strip c[0] and c[-1];
  * after stripping, a *_cm_* block has one entry per Chebyshev order, index n = T_n,
    so c[0] = T_0 = the (profile-weighted, for *_w1) rate in pb;
  * <values> holds 35 numbers = 7 scale choices x 5 PDF slots; only v[0:7] are used,
    in the order (1,1),(2,2),(.5,.5),(1,.5),(1,2),(.5,1),(2,1); central = v[0].
The *_w1 blocks are the profile-weighted sums  W_n = sum_i sigma_i w(pT_i) T_n(u_i),
with w the 30->60 GeV smoothstep (hard cut at 500) and u a log map whose range is
whatever the producer used -- pass it in as (soft_lo, x_hi); it must match the map
the solver builds features on.
"""
import re
import numpy as np

SCALE_ORDER = [(1, 1), (2, 2), (.5, .5), (1, .5), (1, 2), (.5, 1), (2, 1)]


def _blocks(path):
    s = open(path).read()
    out = {}
    for o in re.findall(r"<observable>(.*?)</observable>", s, re.S):
        name = re.search(r"<description>(.*?)</description>", o, re.S).group(1).strip()
        bins = re.findall(r"<bin>(.*?)</bin>", o, re.S)
        V = np.array([[float(x) for x in re.search(r"<values>(.*?)</values>", b, re.S).group(1).split(",")] for b in bins])
        E = np.array([[float(x) for x in re.search(r"<errors>(.*?)</errors>", b, re.S).group(1).split(",")] for b in bins])
        out.setdefault(name, []).append((V[1:-1], E[1:-1]))   # strip flow slots
    sig = float(re.search(r"<values>\s*([^,<]+)", s).group(1))
    sige = float(re.search(r"<errors>\s*([^,<]+)", s).group(1))
    return out, sig, sige


def load_stripper_recoil(path, sigma_fid_pb, x_match=30.0, x_hi=2500.0, soft_lo=30.0,
                         block="pT_Z_cm_rm1_w1", n_max=None, scale_idx=0):
    """Return the recoil moments dict for one Stripper file.

    sigma_fid_pb : the DY fiducial cross section the window rate is quoted
                   against (R = W_0 / sigma_fid), in pb -- from OUR DY run.
    """
    blocks, sig, sige = _blocks(path)
    V, E = blocks[block][0]                      # (n_orders, 35), (n_orders, 35)
    N = V.shape[0] - 1 if n_max is None else min(n_max, V.shape[0] - 1)
    W = V[:, :7]                                 # (n_orders, 7 scales), central = col 0
    e = E[:, 0]
    W0 = W[0, scale_idx]                        # central value taken at this scale choice
    vals = W[1:N + 1, scale_idx] / W0
    # statistical: propagate independently (no covariance shipped)
    stat = np.sqrt((e[1:N + 1] / W0) ** 2 + (W[1:N + 1, 0] * e[0] / W0 ** 2) ** 2)
    # scale: half-range of the ratio over the 7 scale choices
    ratios = W[1:N + 1, :] / W[0, :]
    scl = 0.5 * (ratios.max(1) - ratios.min(1))
    errs = np.sqrt(stat ** 2 + scl ** 2)
    R = W0 / sigma_fid_pb
    # scale: half-range of the window rate over the 7 scale choices (for combine_moments)
    rate_scale = float(0.5 * (W[0, :].max() - W[0, :].min()) / sigma_fid_pb)
    return dict(window_values=vals, window_errors=errs, stat_errors=stat, scale_errors=scl,
                rate_stat=float(e[0] / sigma_fid_pb), rate_scale=rate_scale,
                wprofile_values=vals, wprofile_rate=float(R), rate=float(R),
                x_match=float(x_match), x_hi=float(x_hi), soft_lo=float(soft_lo),
                _meta=dict(file=path, block=block, W0_pb=float(W0), W0_err_pb=float(e[0]),
                           sigma_file_pb=sig, n_orders=int(N)))


def load_stripper_hist(path, name, sigma_fid_pb, smearing="0", nnlojet_scale_order=True):
    """One direct histogram of a Stripper file as (1/sigma_fid) dsigma/dX on its own edges.

    Returns dict(edges, dens (nbins,), stat (nbins,), members (7, nbins)) with the 7
    scale members in NNLOJET order when `nnlojet_scale_order` (Stripper's (1,.5) and
    (2,1) swapped, cf. NNLOJET_TO_STRIPPER in nnlojet_moments), each member divided by
    the corresponding entry of `sigma_fid_pb` (a 7-vector in NNLOJET order, or a
    scalar used for all).  Values in the file are pb per bin; flow slots stripped.
    The file carries two copies per observable (smearing 0 and 0.1): use "0".
    """
    s = open(path).read()
    for o in re.findall(r"<observable>(.*?)</observable>", s, re.S):
        if (re.search(r"<description>(.*?)</description>", o, re.S).group(1).strip() == name
                and re.search(r"<smearing>(.*?)</smearing>", o).group(1).strip() == smearing):
            edges = np.array([float(x) for x in re.search(r"<edges>(.*?)</edges>", o, re.S).group(1).split(",")])
            bins = re.findall(r"<bin>(.*?)</bin>", o, re.S)
            V = np.array([[float(x) for x in re.search(r"<values>(.*?)</values>", b, re.S).group(1).split(",")] for b in bins])[1:-1]
            E = np.array([[float(x) for x in re.search(r"<errors>(.*?)</errors>", b, re.S).group(1).split(",")] for b in bins])[1:-1]
            break
    else:
        raise KeyError(f"{path}: no histogram {name!r} with smearing {smearing}")
    bw = np.diff(edges)
    order = [0, 1, 2, 6, 4, 5, 3] if nnlojet_scale_order else list(range(7))
    sig = np.broadcast_to(np.asarray(sigma_fid_pb, float), (7,))
    members = np.array([V[:, j] / bw / sig[k] for k, j in enumerate(order)])
    return dict(edges=edges, dens=members[0], stat=E[:, 0] / bw / sig[0], members=members)
