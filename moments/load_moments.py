#!/usr/bin/env python3
r"""Print the released moment tables:  python moments/load_moments.py [dy wp wm aa ggh zj]

The loader itself is `maxent_upgrade.load_moments` (maxent_upgrade/released.py):

    from maxent_upgrade import upgrade, load_moments
    moments, config = load_moments("dy")             # the constraint set of the paper
    result = upgrade(events, moments, config)

This module keeps the 1.0 entry point `from moments.load_moments import load` working
when the repository root is the working directory.
"""
import os
import sys

try:
    from maxent_upgrade.released import load_moments, table, observables, sigma_fid, summary, PROCESSES
except ImportError:                                   # a clone that was not pip-installed
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from maxent_upgrade.released import load_moments, table, observables, sigma_fid, summary, PROCESSES

load = load_moments                                   # the 1.0 name
FILES = PROCESSES

__all__ = ["load", "load_moments", "table", "observables", "sigma_fid", "summary", "FILES"]

if __name__ == "__main__":
    for p in (sys.argv[1:] or list(PROCESSES)):
        print(summary(p))
