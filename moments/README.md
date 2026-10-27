# Fixed-order moments, released for use with your own prior

These tables hold the paper's NNLO Drell-Yan input and the inputs for further processes.
The paper's NNLO Z+jet moments are not included and are available from the authors on
request. The `zj` table is a separate NLO calculation with jet towers. Each JSON
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
- where the numbers come from (NNLOJET run and run card);
- the default constraint set.

Complete scripts:
- `examples/upgrade_with_released_moments.py` (the upgrade);
- `examples/bands_with_released_moments.py` (the scale, bootstrap and window variants).

| table | process | order | runs | default towers | also in the table |
| --- | --- | --- | --- | --- | --- |
| `dy_nnlo_moments` | pp → Z/γ* → ℓ⁺ℓ⁻, 13 TeV, ATLAS fiducial | NNLO | 298 | `mll` (BW, 12), `y_abs` (lin, 12, of which the SNR rule imposes 11 on the paper's prior), `pT_ll` window 30→60 GeV, cut 500 (log, 20) | `abs_costh` (\|cos θ*\|, lin [0, 1], 6), joint `mll_x_abs_costh` (16), lepton-pT reference towers `pT_l1` (log [27, 200], 6) and `pT_l2` (log [27, 150], 6) |
| `w_nnlo_moments_WP`, `_WM` | pp → W± → ℓν, 7 TeV, ATLAS fiducial | NNLO | 40 per charge | `m_w` (BW, 12), `y_abs` (lin, 12), `pt_w` window 25→50 GeV, cut 500 (log, 20) | decay-angle window `ctw_window` (flat for \|cos θ*\| < 0.6, zero beyond 0.7; lin, 6), optional; full-range `ctw_full` (8); joint `pt_w_x_ctw` (16) |
| `aa_nnlo_moments` | pp → γγ, 8 TeV, ATLAS fiducial | NNLO | 40 | `m_aa` (log, 6), `abs_costh_aa` (\|cos θ*\|, lin [0, 1], 6), `pt_aa` window 28→56 GeV, cut 500 (log, 12) | window variants of `pt_aa` (`aa_nnlo_moments_winvar`) |
| `ggh_nnlo_moments` | gg → H (on-shell, heavy-top limit), 13 TeV, inclusive | NNLO | 40 | `y_abs` (lin, 6), `pT_H` window 37→74 GeV, cut 1000 (log, 12) | window variants of `pT_H` (`ggh_nnlo_moments_winvar`) |
| `zj_nlo_moments` | pp → Z(→ℓ⁺ℓ⁻) + jet, 13 TeV, ATLAS fiducial, ≥ 1 jet | NLO | 40 | `mll` (BW, 6), `y_abs` (lin, 6), `ptj1` (12) and `ptj2` (6) windows 30→60 GeV, cut 1000, joint `ptj1_x_ptj2` (16), all windowed | `ptj2` to 8 orders; `pimdphi` window 0.05→0.2 (log, 12; the optional tower imposes 4) |

The default depths and transitions are part of each table
(`default_constraints.max_orders` and `.transition`), so `load_moments(process)` returns
the default constraint set without further arguments. The optional towers are one argument
away: `load_moments("wp", recoil=["pt_w", "ctw_window"])`,
`load_moments("zj", recoil=["ptj1", "ptj2", "pimdphi"])`.

## Transitions

By default a recoil tower enters through the blend of the paper: the solver forms the targets
μ_n = (1 − r) ⟨T_n⟩_{p, 1−w} + r ⟨T_n⟩_{FO, w} from the windowed moments and the window rate
of the table, so the prior's spectrum is kept below the window and fixed order rules above it.
Joint moments are imposed in the windowed form only, and every tower a joint block couples
must be windowed too: `load_moments` returns `transition="windowed"` for every recoil tower
when a joint block with a window is requested, and for the Z+jet towers always, which is the
configuration behind the authors' Z+jet solves (61.7% with the jet towers, 35.9% with the
joint moments, 35.3% with the acoplanarity tower added).

## Matching-window variants

For the Higgs boson and diphoton production the recoil tower is also released with three
other matching windows, the inputs of a matching uncertainty:

| table | nominal (a, b) | `window="A"` | `"B"` | `"C"` | upper edge |
|---|---|---|---|---|---|
| `ggh_nnlo_moments_winvar` (`pT_H`) | (37, 74) | (25, 50) | (50, 100) | (37, 148) | cut at 1000 GeV |
| `aa_nnlo_moments_winvar` (`pt_aa`) | (28, 56) | (19, 38) | (37, 74) | (28, 112) | cut at 500 GeV |

`load_moments(process, window=V)` returns the main table with the recoil tower replaced by
variant V: its moments, its rate and its `profile`, which the solver uses (do not change it).
Every other tower is the main table's, and the seam check is skipped for the variant.
`released.windows(process)` lists the windows. The variants come from dedicated NNLOJET runs
(channels R, RV and RR, seeds 101–140, `runcards/ggh_winvar.run` and `gg_winvar.run`), which
also book the nominal window: it reproduces the main table, with a worst pull of 2.1 (Higgs)
and 0.7 (diphoton) over the twelve orders and window rates of 0.21659 against 0.21661 (Higgs)
and 0.28041 against 0.28070 (diphoton), well within their errors (field `validation`).
Their window rate is W0_V / σ_fid, with σ_fid from the six channels of the main run.

## What your prior must provide

`events` is a dict of equal-length arrays:
- the observables named by the table (the `observable` field of each tower, printed by
  `load_moments.py`);
- the strictly positive prior weights under `weight`.

Both sides must integrate the same phase space. Restrict your events to the table's `fiducial`
selection, with the observables defined as there. For the Higgs boson, use the inclusive table
with an inclusive prior. For Z+jet, keep only events with at least one jet; an event without a
second jet can carry `ptj2 = 0` (any value below the window gives the same result).

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

A prior that lacks the channels the corrections populate pays in effective statistics. Under
the same diphoton input, the authors' merged γγ + γγj prior retains 94.87% of its own effective
count (94.9% in the stored solve), and their Pythia 2 → 2 prior retains 3.95% (`result.effN_retained`;
4.0% in the stored solve). That 2 → 2 solve, like
every diphoton, Higgs and Z+jet solve of the authors, imposes every order `load_moments` returns
(`moment_selection=False`): every booked order, except the Z+jet second-jet and acoplanarity
towers, booked to 8 and 12 orders and imposed to 6 and 4. On the authors' other priors the resolution criterion selects every
booked order anyway; on the 2 → 2 prior it drops the sixth \|cos θ*\| order, and the default
solve retains 4.4%.

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
- the window rate `window_rate` = ⟨w⟩/σ_fid (the mean over the runs of the per-run rate), with
  its two errors;
- `matching_scale_rule`, true where `a` is a recoil matching scale.

**`joint[key]`** carries the joint moments ⟨w_x w_y T_m(u(x)) T_n(u(y))⟩/⟨w_x w_y⟩, keyed
`"m,n"`, with their errors and rate.

**At the top level:**
- `provenance`: the NNLOJET run directory, run name and run card (`../runcards/`), the seeds
  and channels pooled, and `decay_type` for the Higgs boson.
- `default_constraints` lists the default constraint set, with its depths in `max_orders` and
  the towers kept windowed in `transition`.
- `hard_scale_GeV` is the Q of the matching-scale rule α_s(p_T) ln²(Q/p_T) < 0.2.
  `load_moments` uses it to check every recoil window, and prints a warning if one disagrees.
- `scales` gives the seven (μ_R, μ_F) choices in the column order of that production's run
  card (central first; the order differs between run cards).
- `sigma_fid_pb` is the fixed-order cross section of the production inside its selection
  (inclusive for the Higgs boson), with its statistical and scale errors and, where needed, a
  `note`. The loader normalizes nothing; `sigma_fid(process)` returns it, and
  `examples/upgrade_with_released_moments.py` uses it to set the absolute rate of the upgraded
  sample.

**A window-variant table** (`*_winvar.json`) carries `variant_of`, `replaces_tower`, its
`provenance` (with the main run that supplies σ_fid) and `windows[V]` for V = A, B, C, each
shaped like a window tower.

**`<table>_runs.json.gz`** holds `{runs, scales, channels, tags, acc[run][tag][scale]}`: every
accumulator above, summed over the perturbative channels of each independent run, at all seven
scales, at full double precision. A window-variant file adds `reference`, the main run's
`norm_born` per run.
- Pooling all runs reproduces every released value, error and rate, to 10⁻⁸
  (`tests/test_released.py`).
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

Both are pooled over the independent runs as the ratio of summed numerators to summed
denominators. Each window rate is the mean over the runs of the per-run rate ⟨w⟩/σ_fid, the
convention of the authors' solves; a variant's rate is the ratio of its two run means
(W0_V over the variant runs, σ_fid over the main runs). Every table states its conventions in
`provenance.conventions`.

## Caveats

* **Higgs cross section.** `sigma_fid("ggh")` is the inclusive NNLO cross section of the
  production, 39.47 pb (heavy-top limit, μ_R = μ_F = m_H, CT18NNLO). The paper normalizes every
  gluon-fusion sample to the inclusive N3LO cross section of the LHC Higgs Cross Section Working
  Group, 48.58 pb, as the ATLAS analysis it compares with does.
* **The Higgs table is fully inclusive.** Use it with an inclusive prior and cut afterwards.
  The acceptance of any later cut then carries shower accuracy.
* **Z+jet** is computed at μ_R = μ_F = H_T/2 and needs at least one jet. Every tower is
  windowed (above).
* **Reference towers** (lepton p_T for Drell-Yan, the full-range W decay angle) are released
  so that a user can check predictions against fixed order. They are not constraints.
  - The two lepton-p_T towers use the maps compiled into NNLOJET, log [27, 200] and
    log [27, 150].
  - The full-range decay angle has an edge at \|cos θ*\| ≃ 0.78, where the neutrino cut closes
    the Born phase space. Only the recoil and the W width populate the region beyond it, which a
    tower should not impose; the optional `ctw_window` imposes the bulk window instead.
* **The map and the window are part of the target.** The loader reproduces the maps and windows
  compiled into the fixed-order runs; do not change them. A different map converges with clean
  closure onto wrong constraints.
* **Window rates.** Each window rate is the mean over the runs of the per-run rate, the
  convention of the authors' solves, so that the tables reproduce the authors' stored weights
  ([`../REPRODUCE.md`](../REPRODUCE.md)). Pooling the rate as a ratio of sums, like the
  moments, would differ by at most 0.05 statistical standard deviations, except for diphoton
  (0.34), and would move the merged diphoton prior's retained fraction from 94.87% to 94.85%.
  A window variant's rate is the ratio of two run means (variant runs, main runs) either way.

All tables were produced by `export_moments.py` from the NNLOJET productions whose run cards
are in `../runcards/`. What they reproduce on the authors' priors is listed in
[`../REPRODUCE.md`](../REPRODUCE.md).
