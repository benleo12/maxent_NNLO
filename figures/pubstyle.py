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
    })
