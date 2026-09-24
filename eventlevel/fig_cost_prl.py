#!/usr/bin/env python3
r"""PRL Fig. 4: cost to reach an effective sample size, MEASURED numbers only.
Derived from for_james_cost/fig_cost_model.py (James Whitehead's template);
read the provenance there before trusting any number.  Differences here:

  * the fixed cost is the SIXTY-seed moment production the papers now use
    (dy_profile_log30_hi), at the uncontended per-seed rate measured in the
    forty-seed dedicated-node campaign, 28.82 core-h per seed set
    (timing_per_seed.csv: 1152.94 core-h / 40): 3 seeds = 86.5 core-h fix the
    central values, 60 seeds = 1729 core-h as run.  As CHARGED in the shared
    queue the sixty-seed run cost 2484 core-h (timing_per_seed_hi.csv), a 1.44x
    contention factor of the same kind the generator timings exclude.
  * a dotted line for the sample with the NNLO Z+jet recoil, which adds the
    1e5 CPU-hours of the Stripper input (R. Poncelet) to the fixed cost.

Generation (Perlmutter EPYC 7713, one core, process CPU time): MG5 LO 0.90,
POWHEG 0.78, MC@NLO 2.23, MiNNLO_PS 67.73 core-ms; shower 6.1 core-ms for every
sample (like for like; the prior's --shower-vars run cost 10.2, drawn faded);
dilution N/N_eff measured on the showered fiducial samples: MaxEnt 1.030,
POWHEG 1.050, MC@NLO 1.256, MiNNLO_PS 3.670; warm-up measured: POWHEG 38.1 s,
MiNNLO_PS 1607.1 s (MC@NLO's is inside its 2.23).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C
use_pub_style(base=17)

MS = 1e-3
PER_SEED = 1152.94 / 40 * 3600.0            # core-s per seed set, uncontended (40-seed campaign)
C_FIXED_3   = 3 * PER_SEED                  # 86.5 core-h
C_FIXED_60  = 60 * PER_SEED                 # 1729 core-h, the production as run
C_FIXED_CHARGED = 2483.6 * 3600.0           # as charged in the shared queue (not drawn)
C_STRIPPER = 1.0e5 * 3600.0                 # NNLO Z+jet input, as reported
C_WARMUP = {"POWHEG": 38.1, r"MiNNLO$_{\mathrm{PS}}$": 1607.1}
SH_COMMON = 6.1 * MS
SAMPLES = [  # label, generation, shower as spent, N/N_eff, colour, NNLO?, label offset
    ("MaxEnt",                   0.90 * MS, 10.2 * MS, 1.030, C["maxent"],  True,  (104, 22)),
    (r"MiNNLO$_{\mathrm{PS}}$", 67.73 * MS,  6.1 * MS, 3.670, C["minnlo"],  True,  (-16, 12)),
    ("MC@NLO",                    2.23 * MS,  6.1 * MS, 1.256, C["mcatnlo"], False, (62, 34)),
    ("POWHEG",                    0.78 * MS,  6.1 * MS, 1.050, C["powheg"],  False, (104, -16)),
]
BY = {s[0]: s for s in SAMPLES}; MAXENT, MINNLO = BY["MaxEnt"], BY[r"MiNNLO$_{\mathrm{PS}}$"]
N_OURS = 0.972 * 742838
c_event = lambda s: s[1] + SH_COMMON
per_eff = lambda s: c_event(s) * s[3]


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.6, 5.6), gridspec_kw={"wspace": 0.27})
    XLO, XHI = 0.98, 5.2
    for x_, txt in ((1e-2, r"$10$ core-ms"), (3e-2, r"$30$ core-ms"), (1e-1, r"$100$ core-ms"), (3e-1, r"$300$ core-ms")):
        xs = np.geomspace(XLO, XHI, 50); a.plot(xs, x_ / xs, color="0.82", lw=1.2, zorder=0)
        xa = XHI / 1.25; y_lab = x_ / xa
        if 6.0e-3 < y_lab < 0.15:
            a.annotate(txt + r" per eff.\ event", (xa, y_lab), fontsize=11, color="0.45", rotation=-32,
                       rotation_mode="anchor", ha="right", va="bottom", zorder=1)
    for lbl, g, sh, d, col, nnlo, off in sorted(SAMPLES, key=lambda s: s[0] == "MaxEnt"):
        y_ll, y_sp = g + SH_COMMON, g + sh
        if abs(y_sp - y_ll) > 1e-12:
            a.plot([d, d], [y_ll, y_sp], color=col, lw=1.6, alpha=0.55, zorder=6)
            a.plot([d], [y_sp], "o", ms=8, color=col, mfc="white", mew=1.6, alpha=0.9, zorder=6)
            a.annotate("as spent", (d, y_sp), textcoords="offset points", xytext=(11, 4), fontsize=10, color=col, alpha=0.85, zorder=6)
        a.plot(d, y_ll, "o" if nnlo else "s", ms=15 if nnlo else 12, color=col, mfc=col if nnlo else "white", mew=2.4,
               zorder=5 if not nnlo else 7, markeredgecolor="white" if nnlo else col, alpha=0.92 if nnlo else 1.0)
        lead = dict(arrowprops=dict(arrowstyle="-", lw=0.9, color=col, shrinkA=0, shrinkB=7)) if off[0] > 0 else {}
        a.annotate(lbl, (d, y_ll), textcoords="offset points", xytext=off, ha="left" if off[0] > 0 else "right",
                   va="center", fontsize=15, color=col, zorder=8, **lead)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(XLO, XHI); a.set_ylim(5.0e-3, 0.16)
    H = [plt.Line2D([], [], ls="", marker="o", ms=11, color="0.35", label=r"NNLO information"),
         plt.Line2D([], [], ls="", marker="s", ms=10, color="0.35", mfc="white", mew=2.0, label=r"NLO+PS")]
    a.legend(handles=H, loc="lower right", fontsize=12.5, labelspacing=0.35, borderaxespad=0.7)
    a.set_xlabel(r"generated events per effective event,\ \ $N/N_{\rm eff}$")
    a.set_ylabel(r"CPU per generated event\ \ [core-s]")

    ne = np.geomspace(1e4, 3e9, 300)
    for lbl, g, sh, d, col, nnlo, _ in SAMPLES:
        if lbl == "MaxEnt":
            lo = (C_FIXED_3 + ne * per_eff(BY[lbl])) / 3600.0; hi = (C_FIXED_60 + ne * per_eff(BY[lbl])) / 3600.0
            b.fill_between(ne, lo, hi, color=col, alpha=0.25, lw=0, zorder=2)
            b.plot(ne, hi, color=col, lw=3.0, zorder=3, label=r"MaxEnt ($C_{\rm fixed}$ $86$--$1730$ core-h)")
            b.plot(ne, lo, color=col, lw=1.2, zorder=3)
            b.plot(ne, (C_FIXED_60 + C_STRIPPER + ne * per_eff(BY[lbl])) / 3600.0, color=C["maxent_nnlo"], lw=2.0, ls=":",
                   zorder=3, label=r"MaxEnt, NNLO$(Zj)$ recoil ($+10^{5}$ core-h)")
        else:
            b.plot(ne, (C_WARMUP.get(lbl, 0.0) + ne * per_eff(BY[lbl])) / 3600.0, color=col, lw=3.0 if nnlo else 2.0,
                   ls="-" if nnlo else "--", label=lbl + ("" if nnlo else r" (NLO)"))
    b.legend(loc="upper left", fontsize=11.5, labelspacing=0.3, handlelength=1.8, borderaxespad=0.5)
    dm = per_eff(MINNLO) - per_eff(MAXENT); cm = C_WARMUP[r"MiNNLO$_{\mathrm{PS}}$"]
    n_lo, n_hi = (C_FIXED_3 - cm) / dm, (C_FIXED_60 - cm) / dm
    b.axvspan(n_lo, n_hi, color="0.5", alpha=0.13, lw=0, zorder=1)
    b.axvline(N_OURS, color="0.55", lw=1.4, ls="--")
    b.annotate("this work", (N_OURS, 4e4), rotation=90, fontsize=12, color="0.45", ha="right", va="top")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(1e4, 3e9); b.set_ylim(30, 3e7)
    b.set_xlabel(r"target effective sample size,\ \ $N_{\rm eff}$"); b.set_ylabel(r"total CPU\ \ [core-hours]")
    out = os.path.join(HERE, "fig_cost_prl.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); print("wrote", out)
    for lbl, g, sh, d, col, nnlo, _ in SAMPLES:
        print(f"  {lbl:22s} gen {1000*g:6.2f} + shower {1000*SH_COMMON:.1f} core-ms, N/Neff {d:.3f} -> {1000*per_eff(BY[lbl]):7.2f} core-ms per effective event")
    print(f"  MiNNLO/MaxEnt per effective event {per_eff(MINNLO)/per_eff(MAXENT):.2f}x = ME {c_event(MINNLO)/c_event(MAXENT):.2f}x x dilution {MINNLO[3]/MAXENT[3]:.2f}x")
    print(f"  fixed cost: 3 seeds {C_FIXED_3/3600:.1f} core-h, 60 seeds {C_FIXED_60/3600:.0f} core-h (charged {C_FIXED_CHARGED/3600:.0f}); crossover N_eff {n_lo:.3g} to {n_hi:.3g}; this work {N_OURS:.3g}")
    for lbl, fixed in (("MaxEnt", C_FIXED_60), (r"MiNNLO$_{\mathrm{PS}}$", cm)):
        print(f"  total at this work's N_eff: {lbl:22s} {(fixed + N_OURS*per_eff(BY[lbl]))/3600:8.1f} core-h")
    with open(os.path.join(HERE, "fig_cost_prl_stripper.txt"), "w") as f:
        f.write(f"crossover with the NNLO(Zj) recoil fixed cost: N_eff {(C_FIXED_60 + C_STRIPPER - cm)/dm:.3g}\n")
    print(f"  crossover with the Stripper input added: N_eff {(C_FIXED_60 + C_STRIPPER - cm)/dm:.3g}")


if __name__ == "__main__":
    main()
