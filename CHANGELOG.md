# Changelog

## 1.1.0 (2026-10-26)

This release matches the paper, B. Assi, *Maximum-entropy matching: accuracy, transport and
cost*. Two things in 1.0.0 did not:

- **1.0.0 had only the windowed transition.** Its recoil towers imposed w(v) T_n(u) and the
  window rate w(v), with targets r ⟨T_n⟩_{FO, w} and r. The paper's results use the blended
  transition below, now the default.
- **1.0.0 shipped gluon-fusion Higgs moments from an off-shell production.** The NNLOJET run
  behind its `ggh` table (`ggh_moments2`, run card `ggh_moments.run`) used `decay_type = 0`, an
  off-shell Higgs boson, and its normalization did not reproduce the gg → H cross section (the
  table carried no cross section). The table is now regenerated from the on-shell production
  behind the paper.

### The method

- **Blended recoil targets (the default).** For a recoil tower the target of each order is
  μ_n = (1 − r) ⟨T_n⟩_{p, 1−w} + r ⟨T_n⟩_{FO, w}, formed by the solver from the released
  windowed moments and window rate; the measurement functions are the bare T_n(u(v)), clipped
  at the window's lower edge (and at d for a smooth upper edge c → d), so the ratio q/p is flat
  below the matching scale and the tilt crosses the window smoothly. A hard upper cut c adds
  one constraint, the share above c. This is the construction of the code behind the paper,
  ported unchanged.
- **`transition` per recoil tower**: `"blend"` (default) or `"windowed"` (the 1.0.0
  construction), reported in `result.report["windows"]` and by `describe()`.
- **Joint moments are imposed in the windowed form only.** A joint block that couples a recoil
  tower with `transition="blend"` raises a `ValueError`: blended towers are in general
  inconsistent with fixed-order joint moments. `load_moments` returns `transition="windowed"` for the Z+jet towers (the paper's
  configuration, also for the solve without joint moments) and for every recoil tower next to a
  windowed joint block.
- A recoil tower without a `profile` uses the step window [x_match, x_hi) under either
  transition (1.0.0 used a separate composite target for it).
- `closure` and `max_pull` are taken over the imposed moments, as the paper defines them; the
  window shares (a window rate, the share above a cut) are reported in `report["shares"]` and
  by `describe()`.

### The released moments

- `ggh` regenerated from the on-shell NNLOJET production `ggh_moments3` (`decay_type = 3`, run
  card `runcards/ggh_moments3.run`, which replaces `ggh_moments.run`). `sigma_fid("ggh")` is now
  its NNLO cross section, 39.47 pb, with a note that the paper normalizes every gluon-fusion
  sample to the YR4 N3LO cross section, 48.58 pb.
- **Matching-window variants** for gg → H and diphoton: `moments/{ggh,aa}_nnlo_moments_winvar`
  (JSON, CSV and per-run accumulators), from dedicated NNLOJET runs whose run cards
  (`runcards/ggh_winvar.run`, `gg_winvar.run`) and patches (`nnlojet_patch/winvar_*.patch`)
  ship too. `load_moments(process, window="A" | "B" | "C")` returns the main table with the
  recoil tower booked with that window; `released.windows(process)` lists them.
- Every table records its provenance: NNLOJET run directory, run name, run card, seeds and
  channels (and `decay_type` for the Higgs boson).
- Per-run, per-scale accumulators of every table (`<name>_runs.json.gz`, channel sums at full
  double precision), from which `load_moments(process, scale=k)` and
  `load_moments(process, runs=positions)` pool the scale variants and bootstrap replicas of the
  paper's bands.
- Window rates are the mean over the runs of the per-run rate, the convention of the paper's
  solves, so that the tables reproduce the paper's weights; 1.0.0 pooled them as the ratio of
  summed accumulators. The two differ by at most 0.05 statistical standard deviations, except
  for diphoton (0.34). Each table records its conventions in `provenance.conventions`.
- Joint blocks carry their `scale_errors`; window-rate scale errors are the half-range over all
  seven scales.

### The package

- `load_moments`, `sigma_fid` and the tables are part of the package (`maxent_upgrade.released`,
  installed with it); `moments/load_moments.py` remains as a command-line printer and keeps the
  1.0.0 name `load`.
- `upgrade()` checks its inputs and stops or warns with a message that names the problem; it
  also warns when a window's profile does not start where its moments were booked.
- New result fields: `effN_retained` (the fraction of a weighted prior's own effective count),
  `max_pull`, `report["towers"]`, `report["shares"]`, `report["q_over_p_range"]`, and
  `describe()`.
- `compute_fo_moments` honours the window profile of each recoil tower and builds the joint
  blocks of `config["mixed"]`.
- The maps and Chebyshev polynomials live in `maxent_upgrade.basis` (`dy_method` remains as an
  alias).
- `export_moments.py` is self-contained and also exports the window variants.
- `alpha_s` is now the two-loop coupling its docstring describes. In 1.0.0 its two-loop term
  carried an extra factor of alpha_s(m_Z), which left it almost one-loop. `matching_scale` now
  gives 28.0 GeV for Q = m_Z, 25.0 GeV for Q = m_W, 27.7 GeV for Q = 90 GeV and 37.3 GeV for
  Q = m_H (1.0.0 gave 27.9, 24.9, 27.6 and 37.1 GeV). The matching scales of the released
  tables (30, 25, 28 and 37 GeV) and the outcome of every `check_seam` call are unchanged.
- Tests (`tests/`), the flowchart (`docs/`), `REPRODUCE.md` and this changelog are new.

### Removed

- `eventlevel/`, the analysis scripts of earlier drafts (windowed-only construction, the
  off-shell Higgs production, earlier priors), `figures/plot_upgrade.py`,
  `examples/example_dy.py`, and the run cards of superseded productions (`dy_moments_full.run`,
  `gg_moments.run`, `ggh_moments.run`).

## 1.0.0

First public release: `upgrade()` with the windowed transition, the legacy histogram path, the
released moment tables with `moments/load_moments.py`, the NNLOJET patch and run cards.
