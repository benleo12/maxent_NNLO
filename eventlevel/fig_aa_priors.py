#!/usr/bin/env python3
r"""What the prior must cover, shown on the diphoton recoil: two LO+PS priors
for the same NNLO input, side by side.

  v4  Pythia internal q qbar -> aa and the gg box (2->2 only): no qg channel,
      no hard real emission beyond the shower.
  v5  MG5 q qbar -> aa and q g -> aa q (the +1-jet matrix element), CKKW-L
      merged in Pythia, plus the same Pythia gg box.

Left : the recoil pT_aa, unit-normalised, for the two priors (dashed), the
       two upgraded samples (solid) and fixed order (NNLO), with the ratio to
       fixed order.  The v4 prior sits far below fixed order above the seam and
       the tilt must lift it by large factors; the v5 prior already carries
       the qg channel there.
Right: the distribution of the tilt q/p over the events of each prior, with
       the retained fraction of the prior's own effective count (Kish),
       the slicing-independent price of the upgrade.

Inputs: the v4 slice set (default of aa_eventlevel_solve.load_prior) with the
kept weights aa_eventlevel_weights_v4.npz, and the v5 set named in AA_PRIOR_V5
(comma list) with the weights in AA_WEIGHTS_V5 (default
aa_eventlevel_weights_v5.npz).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LS, LW      # noqa: E402
use_pub_style(base=17)
from nnlojet_moments import common_seeds, fo_curve_band   # noqa: E402
import aa_eventlevel_solve as A                    # noqa: E402

GGDIR = os.path.join(os.environ.get("NNLOJET_ROOT", os.path.expanduser("~/nnlojet-v1.0.2")), "gg_moments2")
XM = A.XM
COL = {"v4": "#7f7f7f", "v5": C["maxent"]}
LAB = {"v4": r"$q\bar q\to\gamma\gamma$ + box", "v5": r"merged $\gamma\gamma$+jet + box"}


def kish(w):
    w = np.asarray(w, float); return w.sum() ** 2 / (w ** 2).sum()


def load(tag):
    if tag == "v4":
        os.environ.pop("AA_PRIOR", None)
        wf = "aa_eventlevel_weights_v4.npz"
    else:
        os.environ["AA_PRIOR"] = os.environ["AA_PRIOR_V5"]
        wf = os.environ.get("AA_WEIGHTS_V5", "aa_eventlevel_weights_v5.npz")
    ev = A.load_prior()
    W = np.load(os.path.join(HERE, wf)); idx = W["idx"]
    ev = {k: v[idx] for k, v in ev.items()}
    return ev, W["weights"]


def main():
    e = np.geomspace(4.0, 500.0, 26); bw = np.diff(e); ctr = np.sqrt(e[:-1] * e[1:])
    seeds = common_seeds(GGDIR, A.RUN, A.CH, prefix=A.PREFIX)
    _, _, fd, _, _, fst, mem = fo_curve_band(GGDIR, A.RUN, A.CH, seeds, "pt_aa", edges=e, prefix=A.PREFIX, members=True)
    ok = np.isfinite(fd) & (fd > 0)
    nrm = np.nansum(np.where(ok, fd, np.nan) * bw); fo = np.where(ok, fd, np.nan) / nrm
    mem = np.array([np.where(ok, m, np.nan) for m in mem]); mem = mem / np.nansum(mem * bw, axis=1, keepdims=True)
    fo_lo, fo_hi = np.minimum(fo, np.nanmin(mem, 0)), np.maximum(fo, np.nanmax(mem, 0))

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 7.6), gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07, "wspace": 0.22})
    (a, b), (r, rb) = ax
    rb.axis("off"); b.set_position([b.get_position().x0, r.get_position().y0, b.get_position().width,
                                    a.get_position().y1 - r.get_position().y0])
    a.stairs(fo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"], label=r"fixed order (NNLO)")
    r.fill_between(ctr, np.where(ok, fo_lo / fo, np.nan), np.where(ok, fo_hi / fo, np.nan), color=C["band"], alpha=0.55, step="mid", lw=0)
    r.axhline(1, color="k", lw=0.8)
    summary = []
    for tag in ("v4", "v5"):
        ev, q = load(tag)
        x = ev["pt_aa"]; p = ev["weight"]
        def nd(w):
            h, _ = np.histogram(x, e, weights=w / w.sum()); h = h / bw; s_ = (h * bw).sum(); return h / s_
        hp, hq = nd(p), nd(q)
        a.stairs(hp, e, color=COL[tag], ls="--", lw=2.0, label=f"prior: {LAB[tag]}")
        a.stairs(hq, e, color=COL[tag], lw=3.0, label=r"the same, upgraded")
        r.stairs(np.where(ok, hp / fo, np.nan), e, color=COL[tag], ls="--", lw=1.8)
        r.stairs(np.where(ok, hq / fo, np.nan), e, color=COL[tag], lw=2.4)
        # tilt distribution, per event, weighted by the prior (what the sample "sees")
        t = q / (p / p.sum())
        lt = np.log10(np.clip(t, 1e-4, 1e6))
        hb = np.linspace(-2.5, 3.5, 61)
        b.hist(lt, hb, weights=p / p.sum(), histtype="step", color=COL[tag], lw=2.4,
               label=f"{LAB[tag]}: retained {100*kish(q)/kish(p):.0f}\\%")
        kp, kq = kish(p), kish(q)
        summary.append((tag, len(p), kp, kq, 100 * kq / kp, 100 * np.median(np.abs(hp[ok] / fo[ok] - 1)), 100 * np.median(np.abs(hq[ok] / fo[ok] - 1)),
                        float(p[x > XM].sum() / p.sum()), float(q[x > XM].sum() / q.sum())))
    a.set_xscale("log"); a.set_yscale("log"); r.set_xscale("log")
    a.set_xlim(e[0], e[-1]); r.set_xlim(e[0], e[-1]); r.set_ylim(0.0, 2.5)
    for p_ in (a, r):
        p_.axvline(XM, color=C["seam"], lw=2.0, ls="--")
    a.tick_params(labelbottom=False)
    a.set_title(r"$p_T^{\gamma\gamma}$, two priors for one NNLO input")
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}p_T^{\gamma\gamma}$")
    r.set_ylabel(r"ratio to fixed order"); r.set_xlabel(r"$p_T^{\gamma\gamma}$ [GeV]")
    a.legend(loc="lower left", fontsize=11)
    b.set_yscale("log"); b.set_xlabel(r"$\log_{10}(q_i/p_i)$"); b.set_ylabel(r"fraction of the prior weight")
    b.set_title(r"the tilt each prior needs"); b.axvline(0, color="k", lw=0.8)
    b.legend(loc="upper right", fontsize=11)
    out = os.path.join(HERE, "fig_aa_priors.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    for tag, n, kp, kq, ret, mp, mq, rp, rq in summary:
        print(f"  {tag}: N {n:,}  prior Kish {kp:,.0f} ({100*kp/n:.1f}%)  tilted {kq:,.0f} ({100*kq/n:.2f}%)  retained {ret:.1f}%  |  "
              f"pT_aa vs NNLO median: prior {mp:.1f}%  upgraded {mq:.1f}%  |  window rate above seam: prior {rp:.3f} -> {rq:.3f}")
    print("  ->", out)


if __name__ == "__main__":
    main()
