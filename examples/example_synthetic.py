#!/usr/bin/env python3
"""Zero-dependency demo of maxent_upgrade: synthetic PS+LO prior + synthetic FO histograms.
Runs in seconds, needs no external data. Shows the full API and the moment-SNR selection.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxent_upgrade import upgrade, FOHist

rng = np.random.default_rng(1)
N = 200_000
# toy "color-singlet": Born mass M (Gaussian around 91), rapidity Y, recoil qT (falling)
M  = rng.normal(91.0, 3.0, N).clip(66, 116)
Y  = np.abs(rng.normal(0.0, 1.1, N)).clip(0, 2.4)
qT = rng.exponential(8.0, N).clip(0.5, 200.0)
w  = np.ones(N)
events = dict(mll=M, y_abs=Y, pT_ll=qT, weight=w)

# toy FO histograms at two orders: a mild NNLO/NLO shape change ONLY in the recoil tail
def hist(x, edges, w=None):
    v, _ = np.histogram(x, bins=edges, weights=w); return v / np.diff(edges)
m_edges  = np.linspace(66, 116, 21)
y_edges  = np.linspace(0, 2.4, 13)
qt_edges = np.concatenate([[0.5], np.geomspace(2, 200, 30)])
def fo(edges, x, shape=None):
    v = hist(x, edges)
    if shape is not None: v = v * shape
    err = 0.02 * v
    scales = np.array([v * f for f in (1.0, 1.15, 0.87, 1.1, 0.9, 1.05, 0.95)])
    return FOHist(edges[:-1], edges[1:], v, err, scales=scales)
# NLO and NNLO: identical Born (no resolved shape change), harder recoil tail at NNLO
qc = 0.5*(qt_edges[:-1]+qt_edges[1:])
fo_low  = dict(mll=fo(m_edges, M), y_abs=fo(y_edges, Y), pT_ll=fo(qt_edges, qT))
fo_high = dict(mll=fo(m_edges, M), y_abs=fo(y_edges, Y),
               pT_ll=fo(qt_edges, qT, shape=1.0 + 0.4*(qc/100)))  # 40% lift by 100 GeV

cfg = dict(born={'mll':   {'range': (66., 116.), 'map': 'lin'},
                 'y_abs': {'range': (0., 2.4),   'map': 'lin'}},
           recoil={'pT_ll': {'range': (0.5, 200.), 'map': 'log', 'soft_lo': 0.5}},
           band=False)

result = upgrade(events, fo_low, fo_high, cfg)
print(result.summary())
print("chosen moments (SNR-selected):", result.chosen_moments)
print("recoil SNR spectrum:", [round(float(s),1) for s in result.moment_snr['pT_ll']])
print("Born mll SNR (expect all <1, no resolved shape):",
      [round(float(s),1) for s in result.moment_snr['mll']])
assert (result.weights > 0).all(), "weights must be positive"
print("\nOK: positive-weight upgrade, moment count chosen from FO resolution.")
