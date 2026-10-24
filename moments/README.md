# Fixed-order moments, released for use with your own prior

These tables are the fixed-order input behind every application of the papers. Each JSON
file describes itself, and `maxent_upgrade.load_moments` turns it into the two inputs of
`upgrade()`:

```python
from maxent_upgrade import upgrade, load_moments

moments, config = load_moments("dy")       # the constraint set used in the paper
result = upgrade(events, moments, config)  # events: your prior, see below
print(result.describe())
```

`python moments/load_moments.py [dy wp wm aa ggh zj]` prints, for each table:
- the arrays your `events` dict must carry;
- the maps and windows;
- the constraint set the paper imposed.

Complete scripts:
- `examples/upgrade_with_released_moments.py` (the upgrade);
- `examples/bands_with_released_moments.py` (the scale and bootstrap variants).

| table | process | order | runs | towers imposed in the paper | also in the table |
| --- | --- | --- | --- | --- | --- |
| `dy_nnlo_moments` | pp → Z/γ* → ℓ⁺ℓ⁻, 13 TeV, ATLAS fiducial | NNLO | 298 | `mll` (BW, 12), `y_abs` (lin, 12), `pT_ll` window 30→60 GeV, cut 500 (log, 20 + rate) | `abs_costh` (\|cos θ*\|, lin [0, 1], 6), joint `mll_x_abs_costh` (16), lepton-pT reference towers `pT_l1` (log [27, 200], 6) and `pT_l2` (log [27, 150], 6) |
| `w_nnlo_moments_WP`, `_WM` | pp → W± → ℓν, 7 TeV, ATLAS fiducial | NNLO | 40 per charge | `m_w` (BW, 12), `y_abs` (lin, 12), `pt_w` window 25→50 GeV, cut 500 (log, 20 + rate) | decay-angle window `ctw_window` (flat for \|cos θ*\| < 0.6, zero beyond 0.7; 6 + rate), full-range `ctw_full` (8), joint `pt_w_x_ctw` (16) |
| `aa_nnlo_moments` | pp → γγ, 8 TeV, ATLAS fiducial | NNLO | 40 | `m_aa` (log, 6), `abs_costh_aa` (\|cos θ*\|, lin [0, 1], 6), `pt_aa` window 28→56 GeV, cut 500 (log, 12 + rate) | |
| `ggh_nnlo_moments` | gg → H, 13 TeV, inclusive | NNLO | 40 | `y_abs` (lin, 6), `pT_H` window 37→74 GeV, cut 1000 (log, 12 + rate) | |
| `zj_nlo_moments` | pp → Z(→ℓ⁺ℓ⁻) + jet, 13 TeV, ATLAS fiducial, ≥ 1 jet | NLO | 40 | `mll` (BW, 6), `y_abs` (lin, 6), `ptj1` (12) and `ptj2` (6) windows 30→60 GeV, cut 1000, joint `ptj1_x_ptj2` (16) | `ptj2` to 8 orders; `pimdphi` window 0.05→0.2 (12; the paper imposed 4) |

The depths the paper used, where they are shallower than the table, are part of the
table (`default_constraints.max_orders`), so `load_moments(process)` reproduces the paper's
constraint set without further arguments.

## What your prior must provide

`events` is a dict of equal-length arrays:
- the observables named by the table (the `observable` field of each tower, printed by
  `load_moments.py`);
- the strictly positive prior weights under `weight`.

Both sides must integrate the same phase space. Restrict your events to the table's `fiducial`
selection, with the observables defined as there. For the Higgs, use the inclusive table with
an inclusive prior. For Z+jet, keep only events with at least one jet.

Two definitions matter most:

* **Born-level leptons.** Every lepton observable is defined with the QED final-state
  radiation added back to the lepton (the "Born" lepton, dressed with all its own photons),
  as in a QCD-only fixed-order calculation. Bare leptons make a mass tower spend its
  resolution undoing radiation.
* **Decay angles** are Collins-Soper angles in the boson rest frame.
  - For the Z and for the photon pair, the towers use the absolute value, \|cos θ*\| on
    [0, 1] (`abs_costh`, `abs_costh_aa`), with the Collins-Soper z axis along the pair's
    longitudinal momentum.  Compute
    \|cos θ*\| = \|p1⁺ p2⁻ − p1⁻ p2⁺\| / (m √(m² + p_T²)) with p± = E ± p_z of the two
    leptons (photons), exactly as NNLOJET's `eval_abscosthstCS_ll` (`_gam12`).
  - For the W, the table uses `ctw1 = cos θ* + 1` of the charged lepton, so that the window
    can be written on a positive variable.  This angle is SIGNED: cos θ* =
    sign(p_z^W) [(E_ℓ + p_zℓ)(E_ν − p_zν) − (E_ℓ − p_zℓ)(E_ν + p_zν)] / (m √(m² + p_T²))
    (`eval_costhCS_w`); the full-range reference tower `ctw_full` uses the same angle
    unshifted (`ctw`).

A prior that lacks the channels the corrections populate pays in effective statistics.
Under the same diphoton input:
- a Pythia 2 → 2 prior retains 4.0% of its effective count;
- a merged γγ + γγj prior retains 94.5% (`result.effN_retained`).

## Schema

One schema for every table.

**`towers[key]`** carries:
- `kind`: `born`, `window`, or `reference` (released for validation, never imposed);
- the `observable` name;
- `map`: `lin`, `log`, or `bw` with `bw = [M, Gamma]`;
- `range`, `orders`, `values`;
- `stat_errors`: the run-to-run scatter divided by the square root of the number of runs;
- `scale_errors`: the half-range over the seven-point variation;
- `numerator_tags` and `denominator_tag`: the NNLOJET accumulators the tower is pooled from.

Window towers add:
- `soft_lo`, the lower end of their logarithmic map;
- the `profile` `{a, b, c[, d]}` of the C² window: w = 6t⁵ − 15t⁴ + 10t³ with
  t = clip((ln x − ln a)/(ln b − ln a), 0, 1), a hard cut at `c`, or a C² fall between `c`
  and `d`;
- the window rate `window_rate` = ⟨w⟩/σ_fid, with its two errors;
- `matching_scale_rule`, true where `a` is a recoil matching scale.

**`joint[key]`** carries the mixed moments ⟨w_x w_y T_m(u(x)) T_n(u(y))⟩/⟨w_x w_y⟩, keyed
`"m,n"`, with their errors and rate.

**At the top level:**
- `default_constraints` lists what the paper imposed, with its depths in `max_orders`.
- `hard_scale_GeV` is the Q of the matching-scale rule α_s(p_T) ln²(Q/p_T) < 0.2.
  `load_moments` uses it to check every recoil window, and prints a warning if one disagrees.
- `scales` gives the seven (μ_R, μ_F) choices in the column order of that production's run
  card (central first; the order differs between run cards).
- `sigma_fid_pb` is the fixed-order fiducial cross section with its statistical and scale
  errors. The loader normalizes nothing; `sigma_fid(process)` returns it, and
  `examples/upgrade_with_released_moments.py` uses it to set the absolute rate of the upgraded
  sample.

**`<table>_runs.json.gz`** holds `{runs, scales, channels, tags, acc[run][tag][scale]}`: every
accumulator above, summed over the perturbative channels of each independent run, at all seven
scales (12 significant digits).
- Pooling all runs reproduces every released value, error and rate, to 2 × 10⁻⁹.
- `load_moments(process, scale=k)` pools at another scale choice.
- `load_moments(process, runs=positions)` pools over a resample of the runs, which is a
  bootstrap replica.

## Moment definitions

The maps are:
- u_lin = 2(v − lo)/(hi − lo) − 1;
- u_log = 2(ln v − ln lo)/(ln hi − ln lo) − 1;
- u_BW = 2(t(v) − t(lo))/(t(hi) − t(lo)) − 1, with t(v) = arctan((v² − M²)/(M Γ)).

Values outside the range sit at the edge (u is clipped to [−1, 1]), in the fixed-order code
and in the package alike.

The moments are:
- Born: μ_n = ⟨T_n(u(v))⟩ over the fiducial region;
- window: μ_n = ⟨w T_n(u(x))⟩/⟨w⟩, on the logarithmic map [`soft_lo`, `range[1]`].

Both, and the window rates, are pooled over the independent runs as the ratio of summed
numerators to summed denominators.

## Caveats

* **Higgs normalization.** The `norm_born` accumulator of the Higgs production carries an
  overall normalization that does not reproduce the known inclusive cross section. The shape
  moments are ratios and are unaffected, and the paper uses only those. `sigma_fid_pb` is
  therefore `null` for this table; normalize an upgraded sample to a cross section taken from
  elsewhere.
* **The Higgs table is fully inclusive.** Use it with an inclusive prior and cut afterwards.
  The acceptance of any later cut then carries shower accuracy.
* **Z+jet** is computed at μ_R = μ_F = H_T/2 and needs at least one jet.
  - With the table defaults, on the paper's 934,906-event prior, it reproduces the paper:
    effN 61.7% with the jet towers, 35.9% with the joint moments added.
  - Imposing all 8 orders of `ptj2` instead costs effective statistics (36.5%).
* **Reference towers** (lepton p_T for Drell-Yan, the full-range W decay angle) are released
  so that a user can check predictions against fixed order. They are not constraints.
  - The two lepton-p_T towers use the maps compiled into NNLOJET, log [27, 200] and
    log [27, 150].
  - The paper shows why the full-range decay angle should not be imposed.
* **The map and the window are part of the target.** The loader reproduces the maps and windows
  compiled into the fixed-order runs; do not change them. A different map converges with clean
  closure onto wrong constraints.
* **Window rates.** The paper's own solves averaged the per-run window rates. These tables pool
  them, as they pool everything else. The two differ by at most 0.04 statistical standard
  deviations, except diphoton (0.34). No effective-statistics or closure number moves.

All tables were produced by `export_moments.py` from the NNLOJET productions whose run cards
are in `../runcards/`.

Through `load_moments` on the paper's priors, the tables reproduce the paper's solves:

| table | reproduced |
| --- | --- |
| Drell-Yan | weights to 2 × 10⁻⁵; the six scale variants to 3 × 10⁻⁵ |
| W⁺ | effN and closure |
| W⁻ | effN and closure |
| Higgs | effN 59.0% |
| diphoton | 94.5% of the prior's effective count retained |
| Z+jet | 61.7% / 35.9% |
