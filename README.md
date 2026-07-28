# maxent_upgrade

**Upgrade a parton-shower event sample from one fixed-order (FO) accuracy to the next
by maximum-entropy moment reweighting — without generating a single new event, and
without ever producing a negative weight.**

You bring:

* a **PS+*N*LO** event sample (LO+PS, NLO+PS, and so on), with **positive** weights, and
* your own **fixed-order** prediction, as the **event-level moments** `mu_n` (weighted
  sums over your FO phase-space points) of a handful of observables at order *N+k*,
  each with its Monte-Carlo error,

and `maxent_upgrade` returns **new, strictly-positive per-event weights** that move the
sample to order *N+k* in the moments and observables the FO calculation actually resolves,
while leaving the parton-shower physics untouched everywhere it does not.

The method is process-agnostic for any **color-singlet** final state (Drell–Yan, diphoton,
Higgs, VV, …): all it needs from you is which observable is the Born invariant mass, which
is the rapidity, and which is the recoil transverse momentum.

```python
from maxent_upgrade import upgrade
result = upgrade(events, moments, config)
result.weights     # positive per-event weights at order N+k
result.effN        # effective sample fraction (1.0 = no statistical loss)
result.closure     # worst relative moment closure
result.moment_snr  # per-observable moment signal-to-noise spectrum
```

**The fixed-order input is the moments themselves, computed EVENT LEVEL, not a
histogram.** Your FO calculation already integrates any observable, so it integrates
`T_n(u(x))` directly and hands back the single number

    mu_n = sum_j w_j^FO T_n(u(x_j)) / sum_j w_j^FO

with its genuine Monte-Carlo error `sigma_n`. That number, per Chebyshev order `n`, is
the target. Nothing is re-binned. `sigma_n` is what drives the moment-resolution
selection (a moment is imposed only while `|mu_n^FO - mu_n^prior| / sigma_n > 1`).
If you happen to have your FO events in Python, `compute_fo_moments(fo_events, config,
x_match, x_hi)` builds the `moments` dict for you with the correct maps; otherwise fill
it in from your own FO code (structure below). A histogram-based entry point
`upgrade_from_histograms(events, fo_low, fo_high, config)` also exists as a convenience
for people who only have binned FO output, but it re-bins to estimate the moments and
is therefore the less trustworthy path.

This package is a thin, documented wrapper around a frozen, validated physics engine
(`maxent_match.py`, `dy_method.py`, shipped inside the package). It does not re-implement
any physics; it gives that engine a clean public API, chooses the number of constrained
moments for you from the data, and packages the result.

---

## Accuracy statement (read this first)

> The reweighted sample is accurate to **order *N+k*** in the **constrained moments** and in
> **every observable kinematically determined by them**; it retains the **prior parton-shower
> accuracy everywhere else**; and all weights are **positive by construction**
> (`q = p · exp(λ·φ)` is an exponential tilt of the positive prior, so it can never change sign).

Three consequences worth internalizing:

* We upgrade **shapes and the FO-region rate**, not the total inclusive rate independently:
  everything is treated in the space of moments of the observables you declare.
* The recoil spectrum is only reshaped **above a matching scale `x_match`** that the code
  finds from the FO series itself; below it (the Sudakov/resummation region) the prior
  shower is preserved. This is deliberate — bare FO is not trustworthy there.
* An unconstrained "follower" observable inherits the upgrade **only to the extent that it
  correlates with the constrained degrees of freedom**. It is not independently corrected.

---

## How the moments are built (Chebyshev construction)

An observable `x` on a range `[lo, hi]` is mapped to `v` in `[-1, 1]`,

    linear (mass, rapidity):      v = 2 (x - lo)/(hi - lo) - 1
    logarithmic (transverse pT):  v = 2 (ln x - ln lo)/(ln hi - ln lo) - 1

the Chebyshev polynomials follow the recurrence

    T_0 = 1,   T_1 = v,   T_n = 2 v T_{n-1} - T_{n-2},

and the constrained moment is the event-level weighted sum

    mu_n = sum_j w_j T_n(v(x_j)) / sum_j w_j .

For the fixed-order TARGET this sum runs over your FO phase-space points; for closure it
runs over the reweighted prior. It is never taken from a histogram. `chebyshev_moment(x,
w, n_max, lo, hi, map)` is exactly this sum.

### Which observables, at which order (state this precisely)

The recoil is subtle. From an INCLUSIVE color-singlet calculation at NNLO, the singlet
mass and rapidity are NNLO, but the singlet transverse momentum is only NLO(V+jet): a
nonzero recoil needs a real emission. To constrain the recoil at NNLO, take it from the
V+jet calculation at NNLO above a jet-resolution cut. Label each moment by the fixed-order
calculation it comes from, not by the inclusive order of the process. See
`MOMENT_CONSTRUCTION.md` at the repository root for the full recipe.

---

## Installation

Pure Python + NumPy. Copy the `maxent_upgrade/` directory next to your analysis and

```python
import sys; sys.path.insert(0, "/path/to/parent/of/maxent_upgrade")
from maxent_upgrade import upgrade, moment_snr, fo_from_dat, FOHist
```

The package is self-contained: `maxent_match.py` and `dy_method.py` travel with it.

---

## (a) The event arrays you pass in — `events`

`events` is a **dict of 1-D NumPy arrays, all the same length** (one entry per event). It must contain:

| key | what it is | required? |
|-----|------------|-----------|
| `weight` (name configurable via `weight_key`) | the **positive** prior weights of your PS+*N*LO sample | **yes** |
| the **Born** observable arrays | the color-singlet invariant mass and rapidity, e.g. `mll`, `y_abs` | **yes** — the ones you name in `config['born']` |
| the **recoil** observable array | the color-singlet transverse momentum, e.g. `pT_ll` | **yes** — the one you name in `config['recoil']` |
| any number of **follower** arrays | e.g. `phistar`, `pT_lead`, `Δφ`, … | optional |

Notes:

* Weights must be **strictly positive** (a positive prior is the whole point — that is what
  makes the exponential tilt positivity-preserving). If your prior has negative weights,
  fold them in first (rebin/positive-resample) before calling `upgrade`.
* Use whatever variable you like for rapidity as long as it matches your FO histogram
  (`|y|` on `[0, y_max]` or signed `y` on `[-y_max, y_max]`).
* Arrays are used verbatim; no cuts are applied inside the package. Put your fiducial cuts
  on the sample **before** calling, and use the **same cuts** for the FO histograms (see (b)).

---

## (b) The moments you provide — `moments`  (primary interface)

Computed **event level** in your FO code, one weighted sum per Chebyshev order `n`:

```python
moments = {
  'born':   {'mll':   {'values': [mu_1, mu_2, ...], 'errors': [sig_1, sig_2, ...]},
             'y_abs': {'values': [...],             'errors': [...]}},
  'recoil': {'pT_ll': {'window_values': [mu_1, ...], 'window_errors': [sig_1, ...],
                       'rate': R, 'x_match': xm, 'x_hi': xh, 'soft_lo': s0}},
}
```

* `born[obs].values[n-1]` = `sum_j w_j T_n(u(x_j)) / sum_j w_j` over your order-`N+k` FO
  events, full fiducial, with `u = umap(x, a, b, map)` for that observable (same `a,b,map`
  you put in `config`).
* `recoil[obs].window_values[n-1]` = the same sum but **only over FO events with
  `x_match <= pT < x_hi`**, using `u = umap(pT, soft_lo, x_hi, 'log')`.
* `recoil[obs].rate` = FO cross-section fraction in `[x_match, x_hi]`.
* `errors` / `window_errors` = the FO Monte-Carlo uncertainty on each moment (scale
  envelope in quadrature with statistics). These drive the moment-resolution selection.

Helper `compute_fo_moments(fo_events, config, x_match, x_hi)` fills this in for you if your
FO phase-space points are in Python (pass `weight` and optionally `weight_scales`). See
`examples/example_moments.py`.

---

## (b-alt) Histogram fallback — `fo_low`, `fo_high`  (less trustworthy)

Only if you have binned FO output rather than event-level moments. `upgrade_from_histograms`
re-bins to estimate the moments, which introduces binning dependence; prefer (b) above.

## (b-alt continued) (b) The fixed-order histograms you must provide — `fo_low`, `fo_high`

This is the only real work on your side. You must supply, **at the same fiducial cuts as your
events**, the FO differential distributions at **two consecutive orders**:

* `fo_low`  = order **N**   (e.g. NLO)
* `fo_high` = order **N+k** (e.g. NNLO; the example uses k = 1)

for the following observables:

| observable | role | histogram you must provide |
|------------|------|-----------------------------|
| color-singlet **invariant mass** `m` | Born | `dσ/dm` at orders N and N+k |
| color-singlet **rapidity** `y` (or `|y|`) | Born | `dσ/dy` at orders N and N+k |
| color-singlet **transverse momentum** `pT` | recoil | `dσ/dpT` at orders N and N+k |

Each is passed as a `FOHist`:

```python
from maxent_upgrade import FOHist
h = FOHist(
    bin_low  = ...,   # 1-D array of lower bin edges
    bin_high = ...,   # 1-D array of upper bin edges
    value    = ...,   # 1-D array, central-scale dσ/dX per bin (any consistent units)
    error    = ...,   # 1-D array, MC statistical error per bin
    scales   = ...,   # OPTIONAL (n_scale, n_bin): scale-variation rows; row 0 == central
    scale_errors = ...# OPTIONAL (n_scale, n_bin): MC error per scale row
)
fo_high = {"mll": h_m_high, "y_abs": h_y_high, "pT_ll": h_pt_high}
fo_low  = {"mll": h_m_low,  "y_abs": h_y_low,  "pT_ll": h_pt_low}
```

Requirements and conventions:

* **Same cuts as the events.** The FO histograms must be computed with the same fiducial
  selection your event sample lives in, or the differential K-factor is meaningless.
* **Densities or integrals — either is fine**; only shapes and the ratio of the two orders are used.
* **Provide the scale-variation columns** (`scales`, e.g. the 7-point set, row 0 = central).
  These are what give you (i) the FO uncertainty band on the answer and (ii) the denominator
  of the moment signal-to-noise selection (see (e)). Without them you still get a central
  answer, but no band and the SNR selector sees only the FO statistical error.
* The recoil histogram may be **tiled from several files** covering different pT ranges;
  concatenate them into one `FOHist` (the NNLOJET reader below does this).
* Bin edges may differ between observables, and between the recoil sub-ranges; they only
  have to be internally consistent (the code checks that the event-side edges match).

**NNLOJET `.dat` convenience reader.** If your FO comes from NNLOJET-style density files
(`lo center hi  val₁ err₁  val₂ err₂  …  val₇ err₇`, scale 1 = central):

```python
from maxent_upgrade import fo_from_dat
h_pt_high = fo_from_dat(["nnlo.ptz_fine.dat", "nnlo.ptz_mid.dat", "nnlo.ptz_high.dat"])
```

---

## (c) Followers need **no** FO input

Any array in `events` that you list under `config['followers']` (or simply leave out of
`born`/`recoil`) is an **unconstrained follower**. You do **not** provide any FO histogram
for it. It automatically inherits the new per-event weights, so its distribution is upgraded
exactly as far as it correlates with the constrained degrees of freedom — no more, no less.
This is how you read off, e.g., `φ*_η`, the leading-lepton pT, or `Δφ` "for free."

---

## (d) What accuracy you get

See the boxed statement at the top. In one line:

> **N^kLO in the constrained moments and in the observables kinematically determined by them;
> prior shower accuracy everywhere else; positive weights by construction.**

---

## (e) The moment-SNR selection rule (how the number of moments is chosen)

Instead of hand-picking "constrain 6 moments of the mass," the package decides **how many
moments of each observable the FO calculation actually resolves**, from the FO uncertainties
you supplied. For every observable and moment order *n* it forms a **signal-to-noise ratio**

```
SNR_n = | μ_n(target) − μ_n(prior) | / σ_FO(μ_n)
```

where

* `μ_n` is the *n*-th Chebyshev moment on the observable's variable map,
* `μ_n(target)` is the prior moment after applying the binned differential K-factor
  `K(x) = dens_{N+k}(x)/dens_N(x)` (for the recoil, evaluated inside the FO window),
* `σ_FO(μ_n)` is propagated by Monte Carlo from **both** FO uncertainty sources — the
  **scale envelope** (half-spread of `μ_n` over the scale rows) and the **FO statistical
  errors** (resample every bin of both orders within its error, recompute the moment) —
  added in quadrature.

The selection rule (default): **constrain moments up to the largest order *n* with
`SNR_n > threshold` (default `threshold = 1.0`)**, per observable. A moment with `SNR_n < 1`
is one the FO calculation cannot pin down beyond its own uncertainty, so constraining it only
injects FO noise into the weights; it is dropped. The chosen counts are recorded in
`result.chosen_moments` and the full spectra in `result.moment_snr`.

This is the systematic replacement for fixed, hand-tuned moment counts. You can inspect the
spectrum directly:

```python
from maxent_upgrade import moment_snr, resolved_order
snr = moment_snr(fo_low["mll"], fo_high["mll"], events, "mll", 66., 116., "lin", Nmax=10)
n_constrain = resolved_order(snr, threshold=1.0)
```

(In the shipped DY example this rule finds that at NNLO-vs-NLO the Born mass and rapidity
shapes are **not** resolved beyond FO uncertainty — every `SNR_n < 1` — while the recoil pT
is resolved through high order. The upgrade therefore constrains essentially only the recoil,
which is the correct physics content of the NNLO correction for this observable set.)

To switch selection off and impose fixed counts instead, set `moment_selection=False` and
optionally `born_N={obs:n}`, `recoil_N=n`.

---

## (f) Knobs and defaults

All knobs are optional keys in `config`; every default lives in `maxent_upgrade.DEFAULTS`.

### Moment selection

| knob | default | meaning |
|------|---------|---------|
| `moment_selection` | `True` | choose #moments per observable from the SNR rule (below). Set `False` to use fixed counts. |
| `snr_threshold` | `1.0` | a moment is "resolved" (kept) when `SNR_n >` this. Raise it to be more conservative. |
| `snr_max_order` | `{'born':10,'recoil':12}` | ceiling of the SNR scan per role (how many moments to even test). |
| `snr_n_stat` | `200` | Monte-Carlo resamples for the FO **statistical** part of `σ_FO`. |
| `snr_seed` | `0` | RNG seed for that resampling (reproducibility). |
| `born_N` | `None` | fixed per-Born-observable moment count, used only when `moment_selection=False`. `None` → engine default (`N_FO=6`). |
| `recoil_N` | `None` | fixed recoil moment count, used only when `moment_selection=False`. `None` → engine default (`N_W=12`). |

### Physics dials (engine `DIALS`; override selectively in `config`)

| knob | default | meaning |
|------|---------|---------|
| `L2` | `1e-4` | **ridge on the convex dual.** Larger = smoother/more regularized λ (worse closure, more robust); smaller = tighter closure (can overfit FO noise). Closure typically tracks `L2`. |
| `WSIG` | `2.0` | **recoil-window admission by significance.** A recoil bin enters the window only if `value/error > WSIG`. A large correction with a large *relative* error is still admitted (its error goes into the band, it is not vetoed) — `WSIG` gates **significance**, not precision. |
| `RELMAX` | `0.15` | **differential-K / x_match validity mask.** FO bins with relative error `> RELMAX` are dropped from the diff-K target and from the `x_match` convergence criterion (which need precise ratios). Masked cross-section fraction is reported. |
| `DELTA` | `0.20` | **plateau tolerance for `x_match`.** The recoil convergence ratio `K_conv = dens_{N+k}/dens_N` has an interior plateau; `x_match` is the lowest scale below it where `|K_conv/plateau − 1| < DELTA` still holds contiguously. |
| `CLIP` | `(0.3, 3.0)` | **clamp on the differential K-factor** applied to Born reweighting, to bound pathological ratio bins. |
| `MINSUP` | `30` | **support guard:** a recoil window bin needs at least this many *unweighted* prior events to be trusted, else it is dropped (and the window is capped at the last supported bin). |
| `RSUP` | `20.0` | **support guard:** a recoil window bin is dropped if its target/prior probability ratio exceeds this (prevents a starved prior bin from being blown up). |

### x_match rule (no free knob — it is data-blind)

The recoil matching scale is chosen automatically:

* If the FO **order *N*** populates the recoil spectrum (two orders available, `k ≥ 1`), it is
  the **max** of two data-blind bounds: the **series-convergence** bound (plateau of
  `K_conv`) and the **FO-vs-shower seam** bound. The common logarithmic divergence cancels
  in the order ratio but is caught by the shower seam, so both are needed.
* If only the higher order populates the recoil (`k = 1` from a Born-only prior), it is the
  **minimal-discrepancy seam** between the bare FO and the LL shower.

`result.x_match` reports the chosen edge and `result.report['kconv']['rule']` the rule used.

### Rate scheme and the band

The recoil-window **rate** (the probability mass placed in the FO region) is, by default,
`max(prior_rate, FO_positive_part_rate)`; the alternatives `prior` and `fo` are enumerated
into the **matching-scheme band**. With `band=True` (default) `upgrade` also re-solves every
FO **scale** variation, returning all variant weight vectors in `result.band` so you can draw
the uncertainty envelope. Set `band=False` for just the central solve (much faster).

| knob | default | meaning |
|------|---------|---------|
| `band` | `True` | also solve every scale variation + rate scheme; fills `result.band`. |
| `weight_key` | `'weight'` | the key in `events` holding the positive prior weights. |

---

## API reference

### `upgrade(events, fo_low, fo_high, config) -> UpgradeResult`

The one entry point. `UpgradeResult` fields:

| field | type | meaning |
|-------|------|---------|
| `.weights` | `np.ndarray` | new positive per-event weights (central solve). |
| `.effN` | `float` | effective sample **fraction** in (0, 1]; `1.0` = no statistical loss. |
| `.closure` | `float` | worst relative moment closure, `max_i |achieved_i/target_i − 1|`. |
| `.x_match` | `float` | recoil matching scale chosen by the engine. |
| `.report` | `dict` | full engine report (diff-K masks, window, rate, solve stats) plus a `moment_selection` block with the SNR spectra and chosen counts. |
| `.moment_snr` | `dict{obs: np.ndarray}` | per-observable SNR spectrum (orders 1…Nmax). |
| `.chosen_moments` | `dict{obs: int}` | number of moments actually imposed per observable. |
| `.band` | `dict{tag: np.ndarray}` or `None` | variant weight vectors (scales + rate schemes) for the uncertainty envelope. |

`.summary()` returns a one-line human-readable digest.

### `moment_snr(fo_low, fo_high, events, obs, a, b, mp, Nmax, weight_key='weight', window=None, n_stat=200, seed=0) -> np.ndarray`

The per-moment SNR spectrum for one observable (see (e)). `a, b` are the physical range,
`mp ∈ {'lin','log'}` the variable map, `window=(x_lo, x_hi)` restricts to the recoil FO region.

### `resolved_order(snr, threshold=1.0) -> int`

Largest moment order with `SNR_n > threshold` (0 if none) — the selection rule.

### `fo_from_dat(paths, nscales=7) -> FOHist`

Read (and concatenate) NNLOJET-style density `.dat` files into a `FOHist` with all scale rows.

### `FOHist(bin_low, bin_high, value, error, scales=None, scale_errors=None)`

The FO histogram container (see (b)).

---

## Runnable example

`example_dy.py` upgrades a frozen LO+PS Drell–Yan sample (`dy_prior_atlas_v2.npz`, ~2.79 M
positive-weight events) to **NNLO** using NLO and NNLO NNLOJET histograms, and prints the SNR
spectra, `effN`, `x_match`, the chosen moment counts, and the closure:

```bash
python maxent_upgrade/example_dy.py
```

It runs end to end and lands on a **positive-weight** solution with `effN ≈ 98–99 %` and
`closure ≈ 10⁻⁴`, matching the frozen DY NNLO reference.

## What the engine does, in three lines

1. **Born** invariant mass and rapidity → **differential K-factor moment** targets (with
   two-sided validity masks that preserve any FO bin that is statistically noisy or where the
   FO series is breaking down).
2. **Recoil** pT → a **composite window**: FO-shaped above the data-blind `x_match`, prior
   Sudakov shape below it, glued by the FO-region rate.
3. **One convex MaxEnt solve** (z-scored Newton on the dual, ridge `L2`) → positive weights;
   re-solved per scale/rate variation for the band.
