#!/usr/bin/env python3
r"""Shared publication plot style: real LaTeX, large fonts, journal look.

    from pubstyle import use_pub_style
    use_pub_style()

Standalone figures (one plot per file), Computer Modern via usetex, inward ticks
on all four sides, minor ticks, generous font sizes for PRL-column figures.
"""
import matplotlib as mpl


def use_pub_style(base=20):
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base + 2,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "legend.fontsize": base - 4,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.4,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 7, "ytick.major.size": 7,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "xtick.major.width": 1.1, "ytick.major.width": 1.1,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "errorbar.capsize": 2.5,
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# ONE palette and ONE line style per series, used by EVERY figure in both
# papers.  Import these rather than writing colours inline, so a reader who
# learns the colours on one figure can read all of them.
#
#   data        black points        the measurement
#   prior       grey dashed         PS+LO, what we start from
#   maxent      red, thickest       this work
#   fo          black dotted        fixed order, drawn faded where it is not
#                                   a prediction (below the matching scale)
#   minnlo / mcatnlo / powheg       the matched generators
# ---------------------------------------------------------------------------
C = {
    "data":    "k",
    "prior":   "0.55",
    "maxent":  "#d62728",
    "fo":      "k",
    "minnlo":  "#1f77b4",
    "mcatnlo": "#2ca02c",
    "powheg":  "#9467bd",
    "seam":    "0.35",
    "band":    "0.75",
}
LS = {"data": "none", "prior": "--", "maxent": "-", "fo": ":",
      "minnlo": "-", "mcatnlo": "-", "powheg": "-"}
LW = {"prior": 2.0, "maxent": 3.2, "fo": 2.4, "minnlo": 1.9,
      "mcatnlo": 1.9, "powheg": 1.9}
LAB = {"data": None, "prior": r"PS+LO prior", "maxent": r"MaxEnt",
       "fo": r"fixed order", "minnlo": r"MiNNLO", "mcatnlo": r"MC@NLO",
       "powheg": r"POWHEG"}
# fraction of negative weights each generator carries (0 for ours, by construction)
NEG = {"maxent": 0, "minnlo": 23, "mcatnlo": 5, "powheg": 1}
FADE = 0.22        # alpha for fixed order outside its region of validity


def series(ax, key, edges, h, label=None, mask=None, **kw):
    """Draw one series in the shared style.  `h` is a per-bin density."""
    import numpy as np
    y = h if mask is None else np.where(mask, h, np.nan)
    return ax.stairs(y, edges, color=C[key], ls=LS[key], lw=LW.get(key, 2.0),
                     label=label, **kw)


def seam_line(axes, x, label=None):
    """Vertical matching-scale marker, identical on every figure."""
    for a in np.atleast_1d(axes):
        a.axvline(x, color=C["seam"], lw=2.0, ls="--", zorder=1)


import numpy as np  # noqa: E402  (used by seam_line)


def support_mask(x, w, edges, min_eff=100.0):
    """Bins where the REWEIGHTED effective statistics fall below `min_eff`.

    Reweighting cannot create events: where the prior has no support the result
    is an artefact of a handful of events carrying large weight, not a statement
    about the method.  Judge that by effective statistics per bin,
    N_eff = (sum w)^2 / sum w^2, rather than by eye.
    """
    import numpy as np
    x = np.asarray(x, float); w = np.asarray(w, float)
    idx = np.digitize(x, edges) - 1
    n = len(edges) - 1
    s1 = np.bincount(idx[(idx >= 0) & (idx < n)], weights=w[(idx >= 0) & (idx < n)], minlength=n)
    s2 = np.bincount(idx[(idx >= 0) & (idx < n)], weights=w[(idx >= 0) & (idx < n)] ** 2, minlength=n)
    eff = np.where(s2 > 0, s1 ** 2 / np.maximum(s2, 1e-300), 0.0)
    return eff < min_eff, eff


def shade_unsupported(axes, edges, bad, color="0.5", alpha=0.16):
    """Grey out contiguous runs of unsupported bins on every axis given."""
    import numpy as np
    bad = np.asarray(bad, bool)
    if not bad.any():
        return
    i = 0
    for a in np.atleast_1d(axes):
        pass
    runs = []
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            runs.append((edges[i], edges[j + 1])); i = j + 1
        else:
            i += 1
    for a in np.atleast_1d(axes):
        for lo, hi in runs:
            a.axvspan(lo, hi, color=color, alpha=alpha, zorder=0)
