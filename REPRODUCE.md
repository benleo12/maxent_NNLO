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
stored them, its weights. Effective statistics are N_eff/N, or for the sliced diphoton priors
the retained fraction N_eff/N_eff^p.

| solve | `load_moments` call | paper | this release | max \|q/q_paper − 1\| |
|---|---|---|---|---|
| Drell-Yan | `"dy"` | 99.9%; tilts in [0.97, 1.79] | 99.925%; [0.968, 1.789]; closure 9.9 × 10⁻⁵; max pull 0.041 | 1.7 × 10⁻¹⁴ |
| Drell-Yan, six scale re-solves | `"dy", scale=k` | weights reproduced to 3 × 10⁻⁵ | 99.79–99.96% | ≤ 2.3 × 10⁻¹³; scale 4: 6.9 × 10⁻⁸ |
| Drell-Yan, 20 bootstrap re-solves | `"dy", runs=...` | | | ≤ 2.0 × 10⁻¹³ |
| Drell-Yan, linear mass map | `"dy"`, `mll` map `"lin"` | 10.8% | 10.80% | |
| W+ | `"wp"` | 99.8% | 99.762%; closure 5.2 × 10⁻⁴; max pull 0.026 | 5.2 × 10⁻¹⁴; scale re-solves ≤ 2.4 × 10⁻¹²; bootstrap ≤ 1.3 × 10⁻⁶ |
| W+, decay-angle window | `"wp", recoil=["pt_w", "ctw_window"]` | 99.7% | 99.722% | 1.5 × 10⁻¹³; scale ≤ 4.2 × 10⁻¹³; bootstrap ≤ 2.1 × 10⁻⁷ |
| W- | `"wm"` | 99.8% | 99.750%; closure 5.2 × 10⁻⁴; max pull 0.024 | 1.2 × 10⁻¹³; scale ≤ 1.2 × 10⁻⁷; bootstrap ≤ 1.5 × 10⁻⁶ |
| W-, decay-angle window | `"wm", recoil=["pt_w", "ctw_window"]` | 99.7% | 99.729% | 2.5 × 10⁻¹²; scale ≤ 7.5 × 10⁻¹³; bootstrap ≤ 1.3 × 10⁻⁶ |
| gg → H | `"ggh"` | 88.2%; tilts 0.27–2.0 | 88.207%; [0.269, 2.013] | 2.6 × 10⁻¹²; scale ≤ 1.1 × 10⁻¹¹ (scale 6: 1.9 × 10⁻⁷); bootstrap ≤ 9.8 × 10⁻¹² |
| gg → H, windows A, B, C | `"ggh", window=V` | matching band | 84.43%, 91.13%, 92.30% | 0 (bit for bit) |
| diphoton, merged prior | `"aa"` | 94.9%; tilts 0.54–2.6 | 94.868%; [0.541, 2.615]; max pull 0.013 | 0 (bit for bit) |
| diphoton, windows A, B, C | `"aa", window=V` | matching band | 94.54%, 94.91%, 94.91% | 0 (bit for bit) |
| diphoton, 2 → 2 prior | `"aa"`, `moment_selection=False` | 4.0%; tilts up to 3 × 10⁴ | 3.951%; up to 3.1 × 10⁴ | 1.5 × 10⁻¹¹ |
| Z+jet, jet towers | `"zj", mixed=[]` | 61.7% | 61.664% (the paper's stored value to 4 × 10⁻¹⁴) | no stored weights |
| Z+jet, + 16 joint moments | `"zj"` | 35.9%; closure 6 × 10⁻² | 35.876% (to 1.4 × 10⁻¹³); 6.12 × 10⁻² | no stored weights |
| Z+jet, + acoplanarity tower | `"zj", recoil=["ptj1", "ptj2", "pimdphi"]` | 35.3% | 35.335%; 6.20 × 10⁻² | no stored weights |

For Z+jet, the sixteen joint moments reached by the two solves agree with the paper's stored
values to 10⁻¹² of their uncertainty: up to 3.74σ from fixed order with the jet towers alone
(3.7σ in the paper) and within 0.41σ with the joint moments imposed (0.4σ). For Drell-Yan the
standard deviation of ln(q/p) is 0.025, as in the paper. The closures of all these solves are
below 10⁻³ except for Z+jet. Closure and max pull are taken over the moments; `describe()`
prints the window shares separately (the share above the Drell-Yan recoil cut, for instance, is
met to 0.78 of its Monte Carlo error).

**Re-solves.** The targets of the scale and bootstrap re-solves, pooled from the per-run files,
are bit-identical to those of the paper's own solves (checked for the worst cases above). The
re-solves start from the central multipliers (`lam0`) and hold its orders (`fixed_orders`), as
in `examples/bands_with_released_moments.py`; the bootstrap replicas of the paper drew
`np.random.default_rng(1).choice(n_runs, n_runs)` positions per replica. Where a re-solve
differs from the paper's stored band weights by more than 10⁻¹¹, the difference is the solver's
own stopping tolerance (a dual gradient below 10⁻⁶ in standardized units): the same targets
solved from another starting point differ by as much (1.9 × 10⁻⁷ for the Higgs scale 6,
6.6 × 10⁻⁸ for the Drell-Yan scale 4), and both solutions meet the constraints equally well.

**Window rates.** Each table takes a window rate as the mean over the runs of the per-run rate,
the convention of the paper's solves, and pools the moments as the ratio of summed
accumulators (`provenance.conventions` in each table). Pooling the rate too would change no
other number at its printed precision, but would move the merged diphoton prior from 94.87% to
94.85%. A window variant's rate is the ratio of two run means (variant runs, main runs).

**Orders imposed.** The paper's diphoton, Higgs and Z+jet solves imposed every booked order
(`moment_selection=False`). On the paper's priors the default resolution criterion selects
every booked order anyway, except on the 2 → 2 diphoton prior, where it drops the sixth
\|cos θ*\| order (the default solve then retains 4.4%).

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
