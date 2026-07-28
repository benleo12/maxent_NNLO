#!/usr/bin/env python3
"""Diphoton (gamma gamma) config for maxent_upgrade -- the template the diphoton group uses.

Shows the observable roles and MAPS for a WIDE-continuum process: the diphoton mass spans
a decade so it takes the LOG map (unlike the narrow Drell-Yan Z window, which is linear).
Uses synthetic events so it runs with no external data; swap in your own PS+LO gamma gamma
events and your NNLOJET gamma gamma + (jet) moments to do the real upgrade.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from maxent_upgrade import upgrade, compute_fo_moments

rng = np.random.default_rng(7)
Np = 120000
# toy diphoton prior: wide mass (log), two photon pTs, rapidity, recoil
prior = dict(
    m_aa   = np.exp(rng.uniform(np.log(80), np.log(700), Np)),   # wide continuum -> LOG map
    y_aa   = np.abs(rng.normal(0, 0.9, Np)).clip(0, 2.37),
    pt_a1  = rng.uniform(40, 200, Np),
    pt_a2  = rng.uniform(30, 150, Np),
    pt_aa  = rng.exponential(10, Np).clip(1.0, 300),
    weight = np.ones(Np),
)
# toy FO events (order N+k) with a mild recoil-tail lift and 7 scale weights
Nf = 100000
fo = dict(
    m_aa  = np.exp(rng.uniform(np.log(80), np.log(700), Nf)),
    y_aa  = np.abs(rng.normal(0, 0.9, Nf)).clip(0, 2.37),
    pt_a1 = rng.uniform(40, 200, Nf),
    pt_a2 = rng.uniform(30, 150, Nf),
    pt_aa = rng.exponential(10, Nf).clip(1.0, 300),
)
w = 1.0 + 0.3*(fo['pt_aa']/80)
fo['weight'] = w
fo['weight_scales'] = np.array([w*f for f in (1.0,1.15,0.88,1.1,0.9,1.05,0.95)])

# NOTE the MAPS: m_aa is 'log' (wide continuum), rapidity 'lin', pT recoil 'log'.
# Photon pTs pt_a1/pt_a2 are Born-level here (inclusive gamma gamma), 'lin' over their band.
config = dict(
    born = {
        'm_aa':  {'range': (80., 700.), 'map': 'log'},   # WIDE mass -> log
        'y_aa':  {'range': (0., 2.37),  'map': 'lin'},
        'pt_a1': {'range': (40., 200.), 'map': 'lin'},
        'pt_a2': {'range': (30., 150.), 'map': 'lin'},
    },
    recoil = {'pt_aa': {'range': (1., 300.), 'map': 'log', 'soft_lo': 1.0}},
)
moments = compute_fo_moments(fo, config, x_match=12.0, x_hi=300.0)
r = upgrade(prior, moments, config)
print("DIPHOTON (synthetic) RESULT:", r.summary())
print("chosen moments:", r.chosen_moments)
assert (r.weights > 0).all()
print("OK: diphoton config template runs. Replace synthetic arrays with your gamma gamma"
      " events and NNLOJET moments (see MOMENT_CONSTRUCTION.md).")
