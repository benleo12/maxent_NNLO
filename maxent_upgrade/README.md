# maxent_upgrade (package directory)

The documentation is the repository README (`../README.md`).  Modules:

* `__init__.py` — `upgrade`, `UpgradeResult`, `compute_fo_moments`, `chebyshev_moment`,
  `profile_w`, `matching_scale`, `check_seam`, `combine_moments`, and the legacy
  `upgrade_from_histograms` path.  The blended and windowed recoil transitions live in
  `upgrade`.
* `released.py` — `load_moments`, `sigma_fid`, `windows`, `table`, `window_table`, `summary`
  and the released tables (`moments/`, installed as `maxent_upgrade/moments/`).
* `basis.py` — the variable maps (`lin`, `log`, `bw`) and the Chebyshev polynomials.
* `maxent_match.py` — the convex-dual Newton solver, and the legacy histogram engine.
* `dy_method.py` — compatibility alias of `basis.py` for 1.0 scripts.
