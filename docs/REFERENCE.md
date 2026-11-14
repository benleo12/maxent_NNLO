# API reference

For a first run, start with the [quick start](../README.md) or the [examples](../examples/README.md).

## The inputs in detail

### `events`

A dict of 1-D arrays of equal length, one entry per event. It holds:
- the strictly positive prior weights under `'weight'` (or `config["weight_key"]`);
- one array per observable named in `config`.

Every other array is a follower. `upgrade` does not use it, and it inherits `result.weights`.
No cuts are applied inside the package: prior and moments must integrate the same phase space.

`upgrade` stops with a message that says what to fix on:
- a missing array;
- unequal lengths;
- non-finite values;
- a constrained observable without moments;
- a recoil window that holds no prior event (a units hint);
- an unknown `transition`, or a joint block that couples a blended tower.

It warns when more than 5% of a Born observable lies outside its range (such events are
clipped to the edge on both sides, as in the fixed-order code), when a window's profile does
not start where the moments were booked (`x_match`), and on config keys it does not read.

### `moments`

Each moment is a weighted sum over the fixed-order phase-space points, one number per order,
never a histogram:

```python
moments = {
  "born":   {"mll":   {"values": [mu_1, ...], "errors": [s_1, ...],      # errors: stat (+) scale
                       "stat_errors": [...], "scale_errors": [...]},     # optional, recommended
             "y_abs": {...}},
  "recoil": {"pT_ll": {"window_values": [mu_1, ...], "window_errors": [...],
                       "rate": r, "x_match": 30., "x_hi": 2500., "soft_lo": 30.,
                       "stat_errors": [...], "scale_errors": [...], "rate_stat": ..., "rate_scale": ...}},
  "mixed":  {"ptj1_x_ptj2": {"values": {(1, 1): mu_11, ...}, "errors": {...},    # optional
                             "stat_errors": {...}, "scale_errors": {...}, "rate": r}},
}
```

- **Born:** `values[n-1]` = Σ_j w_j T_n(u(v_j)) / Σ_j w_j over the fiducial region.
- **Recoil:** `window_values[n-1]` = ⟨w T_n(u)⟩ / ⟨w⟩. Here u is the logarithmic map on
  [`soft_lo`, `x_hi`], and w is the window of `config["recoil"][obs]["profile"]` (or the hard
  window [`x_match`, `x_hi`) when no profile is given). The window rate is `rate` = ⟨w⟩ / ⟨1⟩,
  and `x_match` is where the window starts. The solver forms the blended targets from these
  numbers; the fixed order supplies nothing else.
- **Joint:** `values[(m, n)]` = ⟨w_x w_y T_m(u(x)) T_n(u(y))⟩ / ⟨w_x w_y⟩ with its rate
  ⟨w_x w_y⟩ / ⟨1⟩ (w = 1 for an observable without a window).
- **Errors:** `errors` / `window_errors` are the total fixed-order uncertainties, the run-to-run
  Monte Carlo error combined with the seven-point scale half-range. They decide which orders
  are resolved. The two parts are also given separately as `stat_errors` and `scale_errors`.
- **Building the dict yourself:**
  - `compute_fo_moments(fo_events, config)` builds it, joint blocks included, from fixed-order
    events held in Python. Without `n_max` it books 12 orders per Born tower and 20 per recoil
    tower.
  - `chebyshev_moment(x, w, n_max, a, b, mp)` is the moment definition itself.

### `config`

Use the configuration returned by the loader. The default uses the acting range,
for example `[30, 500]` GeV for Drell–Yan recoil. The wider booking configuration
below is shown only to explain the fields.

```python
config = {
  "born":   {"mll":   {"range": (66., 116.), "map": "bw"},              # "lin", "log", "bw" or ("bw", M, Gamma)
             "y_abs": {"range": (0., 2.4),   "map": "lin"}},
  "recoil": {"pT_ll": {"range": (30., 2500.), "map": "log", "soft_lo": 30.,
                       "profile": {"a": 30., "b": 60., "c": 500.}}},    # window: rises a -> b, cut at c (or falls c -> d)
}
```

`load_moments("dy", qt_basis="booked")` returns this booking configuration, with the Breit-Wigner map written out as
`("bw", 91.1876, 2.4952)`. A recoil tower may add `"transition": "windowed"` (the default is
`"blend"`); `load_moments` does so for the Z+jet towers.

| knob | default | meaning |
|---|---|---|
| `moment_selection` | `True` | impose each tower up to its highest order with SNR_n > `snr_threshold`; `False` imposes every supplied order |
| `snr_threshold` | `1.0` | the resolution threshold |
| `fixed_orders` | `None` | `{obs: N}`: impose exactly N orders of a tower; holds the re-solves of a band to the central constraint set |
| `L2` | `1e-4` | ridge on the dual (numerical stabilizer) |
| `lam0` | `None` | warm start from a previous `result.report["lam"]` |
| `keep_features` | `False` | also return the feature matrix `Phi` and prior `p` in `result.report` |
| `soft_constraints` | `False` | impose each moment with a Gaussian penalty of width its error (`soft_errors="stat"` or `"total"`); see `combine_moments` |
| `weight_key` | `"weight"` | the key of the prior weights in `events` |

## Results

| field | meaning |
|---|---|
| `weights` | the new weights q_i > 0, summing to one |
| `effN` | Kish effective sample fraction 1/(N Σ q_i²) |
| `effN_retained` | Kish(q)/Kish(p): the fraction of the prior's own effective count that the upgrade keeps (equal to `effN` for an unweighted prior; the meaningful number for a sliced or weighted one) |
| `closure` | worst relative deviation of an achieved moment from its target |
| `max_pull` | worst \|achieved − target\| of a moment in units of the target's Monte Carlo error |
| `moment_snr` | the resolution spectra SNR_n per tower |
| `chosen_moments` | orders imposed per tower and joint block |
| `x_match` | the matching scale of the first recoil window |
| `report` | multipliers `lam`, targets `mu`, `feature_names`, the per-tower table `towers`, `windows` (with each tower's `transition`), `shares`, `q_over_p_range`, `moment_selection` |

The window shares (the share above a blended window's hard cut, the rate of a windowed tower
or joint block) are constraints but not moments: their closure and pull are in
`report["shares"]`. `result.describe()` prints the per-tower table and the shares;
`result.summary()` prints one line.

## Joint blocks with a shared window rate

A supplied mixed block may contain degree-zero factors, such as `(k, 0)`.
For a window on the other observable, `w(y) T_k(x)` constrains the distribution
of `x` inside that window; an unwindowed `x` tower does not already impose it.
Supply these keys and their fixed-order values explicitly. The generic
`compute_fo_moments` helper constructs the positive-degree products by default.

A windowed mixed block normally adds its own window-rate constraint. Set
`config["mixed"][block]["include_rate"] = False` when precisely the same rate
feature is already imposed elsewhere, for example by a windowed angular tower.
This avoids duplicating a constraint. It changes neither the supplied joint
products nor their targets. The default remains `True` for all existing inputs.
Do not omit a distinct joint-window rate just because its two marginal-window
rates are known: their product is a different measurement function.

## Uncertainty bands

The released inputs can be propagated in one call:

```python
from maxent_upgrade import upgrade_with_uncertainties

ensemble = upgrade_with_uncertainties(events, "wp", n_boot=20, seed=1,
                                      recoil=["pt_w", "ctw_window"])
hist = ensemble.histogram(events["pt_w"], bins=[0, 10, 20, 40, 80, 160])
```

`ensemble.central` is the usual `UpgradeResult`. `ensemble.weights[source]`
holds the unit-area weights, and `ensemble.cross_sections[source]` their rates
in pb. `ensemble.arrays()` produces NPZ-ready **cross-section weights**, including
`weight_upgraded` and `weight_scale`, `weight_stat`, `weight_window`, `weight_pdf`
for available sources. Each variation array has shape `(members, events)` and
excludes the central result. PDF rows are members 1..58 in eigenvector-pair order.
`cross_section_<source>_pb`, labels, source names and methods are saved alongside.
`arrays(unit_area=True)` instead saves every weight vector with sum one.

The histogram returns `central`, `<source>_low`, `<source>_high`, `sample_error`
and `edges`. Defaults are densities in pb per observable unit. `density=False`
returns bin integrals. `unit_area=True` normalizes by the full sample, including
events outside the displayed bins. On a saved output, call
`histogram_bands(values, edges, saved)` or use `maxent-upgrade histogram`.
Histogram each coherent variation **before** taking an envelope or Hessian
combination. Taking per-event up/down envelopes would destroy cancellations.

Use `sources=("scale", "window")` to request a subset. The baseline uses the
nominal window. Sources unavailable for the selected constraints appear in
`ensemble.notes`. The stored weights use about `8*N_events*N_variations` bytes.
The function accepts the same Born, recoil, joint and map selections as the loader.
For the optional NNLO Z+jet recoil input, bootstrap propagation covers only the
Drell–Yan part because no per-run NNLO Z+jet replicas were released.

1. **Scale:** re-solve once per scale choice, `load_moments(process, scale=k)` for k = 1..6.
   The per-bin envelope including the central prediction is the band.
2. **Target statistics:** re-solve once per bootstrap replica of the independent fixed-order
   runs, `load_moments(process, runs=rng.integers(0, n_runs, n_runs))`.
3. **Matching:** re-solve with the transition region moved down and up,
   `load_moments(process, window="down")` and `window="up"`, the threshold of the matching-scale
   rule varied by a factor of two. Their envelope around the nominal solve is the matching
   uncertainty.
4. **PDF** (W± and the Higgs boson): re-solve once per CT18NNLO member,
   `load_moments(process, pdf_member=k)` for k = 1..58. `released.hessian_band` turns the 59
   results (member 0 is the central solve) into the 68% CL band.
5. **Sample statistics:** `sample_error` uses the usual weighted-event variance,
   conditional on the fitted weights. Unit-area histograms include covariance
   with their normalization. This does not resample the prior and repeat its fit.

The helper keeps these sources separate. The paper's error bars combine 2 and 5
in quadrature. Every re-solve uses:
- `fixed_orders` = the central solve's `report["moment_selection"]["chosen"]`, so that it
  imposes the same constraint set;
- `lam0` = the central `report["lam"]`, except with a moved window, whose polynomials can
  differ from the central solve's.

Generator variations can be supplied on the same events as
`prior_weights={"ISR down": weights_down, "ISR up": weights_up}`. These are
replacement **prior** weights, not factors multiplying the posterior. The driver
re-solves the central target for each and adds a shower envelope, keeping the
fixed-order rate. For changes to kinematics or event counts, reweight the separate
samples and compare their histograms. For a new process use `upgrade_ensemble`
with `Variation` inputs, as in [custom_uncertainties.py](../examples/custom_uncertainties.py).
Pass a shifted `config` when maps/windows change; this triggers a cold start.
Custom PDF prescriptions can be applied to member histograms instead of using
the CT18-specific `hessian68` method.

The lower-level [band example](../examples/bands_with_released_moments.py) remains
available to inspect the loops explicitly. Its older file format keeps unit-area
weights and separate `sigma_*_pb` arrays.

Above the highest transition region and below the hard cutoff, all three windows
assign unit weight to the fixed-order input. Their finite-moment posteriors can
still differ there, since each solve also fits the rest of its constrained range.
The moved-window tables also carry independent Monte Carlo uncertainties.

The released tables reproduce the authors' stored weights: the central solves to 2 × 10⁻¹¹ or better,
the moved Higgs and diphoton windows bit for bit, and the scale and bootstrap
re-solves, whose targets are identical to the authors', to between 10⁻¹³ and 1.5 × 10⁻⁶: the
larger differences are the solver's own stopping tolerance. On the paper's priors the central,
moved-window and CT18NNLO-member solves reproduce the stored weights to round-off
([reproduction record](../REPRODUCE.md)).

## Moments from your own fixed-order code

- [`MOMENT_CONSTRUCTION.md`](../MOMENT_CONSTRUCTION.md): the recipe (maps, window, errors, how
  many orders, which calculation each tower comes from).
- [`nnlojet_patch/`](../nnlojet_patch/): the patches that make NNLOJET v1.0.2 accumulate
  event-level moments as single-number profile histograms, and those that add the moved
  windows.
- [`runcards/`](../runcards/): the NNLOJET run cards of every production behind a table.
- [`export_moments.py`](../export_moments.py): pools the runs into the tables
  (`NNLOJET_ROOT=<dir> python export_moments.py dy`).

## Glossary: notation and code

| notation | code |
|---|---|
| prior p_i | `events["weight"]` |
| tower of depth N (moments n = 1..N of one observable) | the lists in `moments`; `n_orders` in a table |
| matching scale v_m | `x_match`; `profile["a"]` |
| window w(v), rising v_m → 2 v_m, cut at c | `config["recoil"][obs]["profile"] = {"a", "b", "c"[, "d"]}`; `profile_w` |
| window rate r = ⟨w⟩ | `rate`, `window_rate` |
| blended target (the default) / windowed form | `config["recoil"][obs]["transition"]` = `"blend"` / `"windowed"` |
| transition region moved down and up (the matching uncertainty) | `load_moments(process, window="down")`, `"up"`; `released.windows` |
| CT18NNLO member k; the PDF band | `load_moments(process, pdf_member=k)`; `released.hessian_band` |
| follower (never constrained) | any array in `events` not named in `config` |
| upgraded sample (MaxEnt in figures) | `result.weights` |
| resolution criterion SNR_n > 1 | `moment_selection`, `snr_threshold`, `result.moment_snr` |
| closure; worst pull | `result.closure`; `result.max_pull` |
| ridge ε of the dual | the config key `L2` |

## Legacy: histogram input

`upgrade_from_histograms(events, fo_low, fo_high, config)` implements an earlier,
histogram-based form of the construction:
- binned K-factor targets from two consecutive orders;
- a data-driven matching scale;
- the dials `WSIG`, `RELMAX`, `DELTA`, `CLIP`, `MINSUP`, `RSUP` and `band`.

It is kept for users who have only binned fixed-order output, and was not used for the paper's
results. `examples/example_synthetic.py` shows it. Its helpers are `moment_snr`,
`resolved_order`, `FOHist` and `fo_from_dat`.
