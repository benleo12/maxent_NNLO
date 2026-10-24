# Reproducing the paper

The repository reproduces two steps of the paper (B. Assi, *Maximum-entropy moment matching for
color-singlet processes*):

1. the released moment tables, from the NNLOJET productions;
2. the paper's solves, from the released tables, given the paper's priors.

It does not distribute the event samples (the priors) or the NNLOJET output files, which are too
large, nor the scripts that draw the paper's figures.

## 1. The tables, from NNLOJET

1. Patch and build NNLOJET v1.0.2 with [`nnlojet_patch/`](nnlojet_patch/README.md) (the two
   `winvar_*` patches only for the window-variant runs).
2. Run every card of [`runcards/`](runcards/README.md) once per perturbative channel and per
   seed. Each table's `provenance` lists the run directory, run card, channels and seeds; the
   run-card README says which runs made the Vegas grids.
3. Pool the runs into the tables:

   ```bash
   NNLOJET_ROOT=<dir with the run directories> NNLOJET_WINVAR_ROOT=<dir with ggh_winvar, gg_winvar> \
       python export_moments.py            # or a subset: dy wp wm aa ggh zj ggh_winvar aa_winvar
   ```

4. `python tests/test_released.py` checks that pooling the per-run accumulators shipped with
   every table reproduces every released number.

## 2. The paper's solves, from the tables

Each row below was re-solved with this release, `upgrade(events, *load_moments(...))`, on the
paper's own prior, and compared with the numbers of the paper and, where the paper's solve
stored them, its weights. "Tables" are the released tables as they are; "paper's rates" replaces
each window rate by the convention of the paper's solves (see the note below the table). Effective
statistics are N_eff/N, or for the sliced diphoton priors the retained fraction N_eff/N_eff^p.

| solve | `load_moments` call | paper | this release (tables) | max \|q/q_paper − 1\|: tables / paper's rates |
|---|---|---|---|---|
| Drell-Yan | `"dy"` | 99.9%; tilts in [0.97, 1.79] | 99.925%; [0.968, 1.789]; closure 9.9 × 10⁻⁵; max pull 0.041 | 1.5 × 10⁻⁵ / 2.9 × 10⁻¹³ |
| Drell-Yan, six scale re-solves | `"dy", scale=k` | weights reproduced to 3 × 10⁻⁵ | 99.79–99.96% | 2.4 × 10⁻⁵ / 8.8 × 10⁻¹¹ |
| Drell-Yan, 20 bootstrap re-solves | `"dy", runs=...` | | | 2.3 × 10⁻⁵ / – |
| Drell-Yan, linear mass map | `"dy"`, `mll` map `"lin"` | 10.8% | 10.80% | |
| W+ | `"wp"` | 99.8% | 99.762%; closure 5.2 × 10⁻⁴; max pull 0.026 | 6.2 × 10⁻⁶ / 1.5 × 10⁻¹² |
| W+, decay-angle window | `"wp", recoil=["pt_w", "ctw_window"]` | 99.7% | 99.722% | 3.1 × 10⁻⁵ / 3.2 × 10⁻¹² |
| W- | `"wm"` | 99.8% | 99.750%; closure 5.2 × 10⁻⁴; max pull 0.024 | 6.2 × 10⁻⁶ / 8.2 × 10⁻¹³ |
| W-, decay-angle window | `"wm", recoil=["pt_w", "ctw_window"]` | 99.7% | 99.729% | 2.8 × 10⁻⁵ / 1.8 × 10⁻¹² |
| W±, six scale re-solves each | `"wp"`, `"wm"`, `scale=k` | | | 4.9 × 10⁻⁵ / – |
| gg → H | `"ggh"` | 88.2%; tilts 0.27–2.0 | 88.207%; [0.269, 2.013] | 8.0 × 10⁻⁸ / 3.6 × 10⁻¹² |
| gg → H, six scale re-solves | `"ggh", scale=k` | | | 2.6 × 10⁻⁷ / – |
| gg → H, windows A, B, C | `"ggh", window=V` | matching band | 84.43%, 91.13%, 92.30% | 2.8 × 10⁻¹¹ (the same in both) |
| diphoton, merged prior | `"aa"` | 94.9%; tilts 0.54–2.6 | **94.85%**; [0.540, 2.617] | 6.3 × 10⁻³ / 7.6 × 10⁻¹³ |
| diphoton, windows A, B, C | `"aa", window=V` | matching band | 94.54%, 94.91%, 94.91% | 1.3 × 10⁻¹² (the same in both) |
| diphoton, 2 → 2 prior | `"aa"`, `moment_selection=False` | 4.0%; tilts up to 3 × 10⁴ | 3.97%; up to 3.1 × 10⁴ | 1.3 × 10⁻² / 2.6 × 10⁻¹² |
| Z+jet, jet towers | `"zj", mixed=[]` | 61.7% | 61.666% | no stored weights |
| Z+jet, + 16 joint moments | `"zj"` | 35.9%; closure 6 × 10⁻² | 35.877%; 6.12 × 10⁻² | no stored weights |
| Z+jet, + acoplanarity tower | `"zj", recoil=["ptj1", "ptj2", "pimdphi"]` | 35.3% | 35.337%; 6.20 × 10⁻² | no stored weights |

For Z+jet, the sixteen joint moments reached by the two solves agree with the paper's to
4 × 10⁻⁴ of their uncertainty: up to 3.74σ from fixed order with the jet towers alone (3.7σ in
the paper) and within 0.41σ with the joint moments imposed (0.4σ). For Drell-Yan the standard
deviation of ln(q/p) is 0.025, as in the paper. The closures of all these solves are below 10⁻³
except for Z+jet. Closure and max pull are taken over the moments; `describe()` prints the
window shares separately (the share above the Drell-Yan recoil cut, for instance, is met to
0.78 of its Monte Carlo error).

**Window rates.** The paper's solves took each window rate as the mean over the runs of the
per-run rate; the tables pool the rate as the ratio of summed accumulators, like every other
number. Under the blend the rate enters every recoil target. Only the merged diphoton prior
moves a quoted number at its printed precision: 94.85% with the tables, 94.87% (94.9%) in the
paper. To use the paper's convention, replace the rate before the solve:

```python
import gzip, json, numpy as np
from maxent_upgrade import upgrade, load_moments

moments, config = load_moments("aa")
r = json.load(gzip.open("moments/aa_nnlo_moments_runs.json.gz", "rt"))
A, tag = np.array(r["acc"]), r["tags"].index
rate = float(np.mean(A[:, tag("prof_wpt_0"), 0] / A[:, tag("norm_born"), 0]))   # central scale
moments["recoil"]["pt_aa"]["rate"] = moments["recoil"]["pt_aa"]["wprofile_rate"] = rate
```

The denominator tag of each window tower is its `denominator_tag` in the table.

**Orders imposed.** The paper's diphoton, Higgs and Z+jet solves imposed every booked order
(`moment_selection=False`). On the paper's priors the default resolution criterion selects
every booked order anyway, except on the 2 → 2 diphoton prior, where it drops the sixth
\|cos θ*\| order (the default solve then retains 4.4%).

**Re-solves.** The scale, bootstrap and window re-solves hold the central solve's orders
(`fixed_orders`) and start from its multipliers (`lam0`), as in
`examples/bands_with_released_moments.py`. The bootstrap replicas of the paper drew
`np.random.default_rng(1).choice(n_runs, n_runs)` positions per replica.

## The priors

| prior | events | used for |
|---|---|---|
| MG5_aMC LO pp → ℓ⁺ℓ⁻ + Pythia 8.312, 13 TeV, ATLAS fiducial selection on Born-level leptons | 747,028 | Drell-Yan |
| MG5_aMC LO pp → μ⁺ν, μ⁻ν̄ + Pythia 8.312, 7 TeV, the NNLOJET W selection | 521,316 (W+), 487,992 (W-) | W± |
| Pythia 8.312 gg → H, 13 TeV, inclusive | 2,000,000 | gg → H |
| merged MG5_aMC γγ + γγj (CKKW-L) with the Pythia gg box, nine slices, 8 TeV, ATLAS selection | 10,644,601 | diphoton (merged) |
| Pythia q q̄ → γγ and gg box, five slices, 8 TeV, ATLAS selection | 6,938,827 | diphoton (2 → 2) |
| MG5_aMC LO pp → ℓ⁺ℓ⁻ + Pythia 8.312, 13 TeV, parton-level jets, ≥ 1 jet | 934,906 | Z+jet |

The paper describes them; they are not distributed with the repository.
