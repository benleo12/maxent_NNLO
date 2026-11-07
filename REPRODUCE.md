# Reproducing the published results

The repository reproduces two steps of the paper (B. Assi, *Maximum-entropy reweighting of
color-singlet samples*), of the letter (B. Assi, R. Poncelet, J. Whitehead, *Positive-weight
moment matching at NNLO and beyond*) and of the authors' further solves:

1. the released moment tables, from the NNLOJET productions;
2. the authors' solves, from the released tables, given the authors' priors.

It does not distribute the event samples (the priors) or the NNLOJET output files, which are too
large, nor the scripts that draw the paper's figures. The event samples are available from the
authors on request.

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

The tables added for the paper come from further NNLOJET runs, which `export_moments.py` does
not pool:
- the window variants of Drell-Yan, W± and Z+jet and the Higgs variant D, booked with the
  nominal window at the central scale (channels R, RV, RR; LO, V, R for Z+jet; seeds from 101);
- the CT18NNLO members of W± and the Higgs boson, one full NNLO run per member at the central
  scale (seeds 101 and 102).

They were built from the per-seed channel sums of these runs. Each table's `provenance` names the
runs, its `conventions` say how they enter, and its `validation` gives the closure of the
nominal window, or of member 0, on the main table. The Z+jet acoplanarity tower `aco` is
integrated from the binned NLO distribution `dphil_a` of the `zj_moments2` runs. Its field
`binned` holds the bin edges and the in-bin averages (`moments/README.md`).

## 2. The letter's solves and the authors' earlier solves, from the tables

These rows are solved with `qt_basis="booked"`, the moments as booked, as the letter and the
releases before 2026-11-06 imposed them (`load_moments(..., qt_basis="booked")`).

Each row below was re-solved with this release, `upgrade(events, *load_moments(...))`, on the
authors' own prior, and compared with the authors' stored numbers and, where the solve stored
them, its weights q_ref. The letter quotes the central Drell-Yan and gg → H values. Effective
statistics are N_eff/N, or for the sliced diphoton priors
the retained fraction N_eff/N_eff^p.

| solve | `load_moments` call | authors' solve | this release | max \|q/q_ref − 1\| |
|---|---|---|---|---|
| Drell-Yan | `"dy"` | 99.9%; tilts in [0.97, 1.79] | 99.925%; [0.968, 1.789]; closure 9.9 × 10⁻⁵; max pull 0.041 | 1.7 × 10⁻¹⁴ |
| Drell-Yan, NNLO Z+jet recoil | `"dy", nnlo_zj=True` | 99.9%; tilts in [0.96, 3.95] | 99.875%; [0.964, 3.955]; closure 9.4 × 10⁻⁵ | 2.6 × 10⁻¹³ |
| Drell-Yan, six scale re-solves | `"dy", scale=k` | weights reproduced to 3 × 10⁻⁵ | 99.79–99.96% | ≤ 2.3 × 10⁻¹³ |
| Drell-Yan, 20 bootstrap re-solves | `"dy", runs=...` | | | ≤ 2.0 × 10⁻¹³ |
| Drell-Yan, linear mass map | `"dy"`, `mll` map `"lin"` | 10.8% | 10.80% | |
| W+ | `"wp"` | 99.8% | 99.762%; closure 5.2 × 10⁻⁴; max pull 0.026 | 5.2 × 10⁻¹⁴; scale re-solves ≤ 2.4 × 10⁻¹²; bootstrap ≤ 1.9 × 10⁻⁷ |
| W+, decay-angle window | `"wp", recoil=["pt_w", "ctw_window"]` | 99.7% | 99.722% | 1.5 × 10⁻¹³; scale ≤ 4.2 × 10⁻¹³; bootstrap ≤ 1.1 × 10⁻⁷ |
| W- | `"wm"` | 99.8% | 99.750%; closure 5.2 × 10⁻⁴; max pull 0.024 | 1.2 × 10⁻¹³; scale ≤ 1.2 × 10⁻⁷; bootstrap ≤ 1.5 × 10⁻⁶ |
| W-, decay-angle window | `"wm", recoil=["pt_w", "ctw_window"]` | 99.7% | 99.729% | 1.5 × 10⁻¹³; scale ≤ 3.0 × 10⁻¹³; bootstrap ≤ 1.3 × 10⁻⁶ |
| gg → H | `"ggh"` | 88.2%; tilts 0.27–2.0 | 88.207%; [0.269, 2.013] | 2.6 × 10⁻¹²; scale ≤ 3.1 × 10⁻¹¹; bootstrap ≤ 9.8 × 10⁻¹² |
| gg → H, windows A, B, C | `"ggh", window=V` | matching band | 84.43%, 91.13%, 92.30% | 0 (bit for bit) |
| diphoton, merged prior | `"aa"` | 94.9%; tilts 0.54–2.6 | 94.868%; [0.541, 2.615]; max pull 0.013 | 0 (bit for bit) |
| diphoton, windows A, B, C | `"aa", window=V` | matching band | 94.54%, 94.91%, 94.91% | 0 (bit for bit) |
| diphoton, 2 → 2 prior | `"aa"`, `moment_selection=False` | 4.0%; tilts up to 3 × 10⁴ | 3.951%; up to 3.1 × 10⁴ | 1.5 × 10⁻¹¹ |
| Z+jet, jet towers | `"zj", mixed=[]` | 61.7% | 61.664% (the stored value to 4 × 10⁻¹⁴) | no stored weights |
| Z+jet, + 16 joint moments | `"zj"` | 35.9%; closure 6 × 10⁻² | 35.876% (to 1.4 × 10⁻¹³); 6.12 × 10⁻² | no stored weights |
| Z+jet, + acoplanarity tower | `"zj", recoil=["ptj1", "ptj2", "pimdphi"]` | 35.3% | 35.335%; 6.20 × 10⁻² | no stored weights |

For Z+jet, the sixteen joint moments reached by the two solves agree with the stored
values to 10⁻¹² of their uncertainty: up to 3.74σ from fixed order with the jet towers alone
(3.7σ in the stored solve) and within 0.41σ with the joint moments imposed (0.4σ). For Drell-Yan the
standard deviation of ln(q/p) is 0.025, as in the stored solve. The closures of all these solves are
below 10⁻³ except for Z+jet. Closure and max pull are taken over the moments; `describe()`
prints the window shares separately (the share above the Drell-Yan recoil cut, for instance, is
met to 0.78 of its Monte Carlo error).

**Drell-Yan cross-moments.** `load_moments("dy", mixed=["mll_x_abs_costh"])` adds the 16 joint
moments of m_ll and \|cos θ*\| of the letter's End Matter. The prior must then also carry
`abs_costh`.

**Re-solves.** The targets of the scale and bootstrap re-solves, pooled from the per-run files,
are bit-identical to those of the authors' own solves (checked for the worst cases above). The
re-solves start from the central multipliers (`lam0`) and hold its orders (`fixed_orders`), as
in `examples/bands_with_released_moments.py`; the authors' bootstrap replicas drew
`np.random.default_rng(1).choice(n_runs, n_runs)` positions per replica. Where a re-solve
differs from the stored band weights by more than 10⁻¹⁰ (some W± bootstrap replicas and the
W- scale 4), the stored solve stopped at the solver's round-off floor: the same targets solved
from a cold start agree with this release to 4 × 10⁻⁸, and both meet the constraints equally
well.

**Window rates.** Each table takes a window rate as the mean over the runs of the per-run rate,
the convention of the authors' solves, and pools the moments as the ratio of summed
accumulators (`provenance.conventions` in each table). Pooling the rate too would change no
other number at its printed precision, but would move the merged diphoton prior from 94.87% to
94.85%. A window variant's rate is the ratio of two run means (variant runs, main runs).

**Orders imposed.** The authors' diphoton, Higgs and Z+jet solves imposed every order
`load_moments` returns (`moment_selection=False`). That is every booked order, except the Z+jet
second-jet and acoplanarity towers, booked to 8 and 12 orders and imposed to 6 and 4. On the authors' priors the default resolution criterion selects
every booked order anyway, except on the 2 → 2 diphoton prior, where it drops the sixth
\|cos θ*\| order (the default solve then retains 4.4%).

## 3. The paper's solves, from the tables

The paper's Drell-Yan, W± and Z+jet solves use new priors (below), with matrix-element
corrections off. Its gg → H and diphoton solves use the priors of section 2. Each row below was
re-solved with this release on the paper's prior, with the configuration of the authors' solve.
- The central solves use the defaults of `load_moments` and `upgrade`. They are compared with
  the authors' stored weights q_ref.
- Each member re-solve holds the central solve's orders and starts from its multipliers
  (`fixed_orders`, `lam0`), as in `examples/bands_with_released_moments.py`. Each window re-solve
  holds the central solve's orders from a cold start (for Z+jet, the orders of the solve with
  the joint moments, plus the four acoplanarity orders).
- The authors stored the weights of these re-solves as per-bin sums only. They are compared bin
  by bin: Drell-Yan p_T^ℓℓ (a fine grid of 600 bins and three coarser ones), m_ll and \|y_ll\|;
  W \|y_W\|, \|η_ℓ\|, m_ℓν, p_T^W and cos θ*; Higgs p_T^H, p_T^j1, \|y_H\| and the jet
  multiplicity; Z+jet p_T^j1, p_T^j2 and the acoplanarity.

The last column is max \|q/q_ref − 1\| over the events, or over the bins for "bins".

| solve | `load_moments` call | N_eff/N, this release (the authors' solve) | closure; max pull | max \|q/q_ref − 1\| |
|---|---|---|---|---|
| Drell-Yan | `"dy"` | 99.913% (99.913%) | 5.1 × 10⁻⁵; 0.039 | 0 |
| Drell-Yan, windows A, B, C | `"dy", window=V` | 99.907, 99.919, 99.922% (the same) | ≤ 5.2 × 10⁻⁵ | ≤ 2.3 × 10⁻⁹ (bins) |
| W+ | `"wp"` | 99.751% (99.751%) | 5.5 × 10⁻⁴; 0.043 | 0 |
| W+, windows A, B, C | `"wp", window=V` | 99.692, 99.761, 99.764% (the same) | ≤ 5.9 × 10⁻⁴ | ≤ 5.8 × 10⁻¹⁰ (bins) |
| W+, CT18NNLO members 1, 29, 58 | `"wp", pdf_member=k` | 99.737, 99.738, 99.758% (the same) | ≤ 6.7 × 10⁻⁴ | 0 (bins) |
| W+, decay-angle window | `"wp", recoil=["pt_w", "ctw_window"]` | 99.707% (99.707%) | 7.0 × 10⁻⁴; 0.044 | 0 |
| the same, windows A, B, C | `..., window=V` | 99.645, 99.719, 99.722% (the same) | ≤ 7.5 × 10⁻⁴ | ≤ 5.6 × 10⁻¹⁰ (bins) |
| the same, members 1, 29, 58 | `..., pdf_member=k` | 99.698, 99.697, 99.715% (the same) | ≤ 8.3 × 10⁻⁴ | 0 (bins) |
| W- | `"wm"` | 99.730% (99.730%) | 5.1 × 10⁻³; 0.039 | 0 |
| W-, windows A, B, C | `"wm", window=V` | 99.682, 99.732, 99.733% (the same) | ≤ 5.1 × 10⁻³ | ≤ 1.8 × 10⁻⁷ (bins) |
| W-, CT18NNLO members 1, 29, 58 | `"wm", pdf_member=k` | 99.716, 99.609, 99.731% (the same) | ≤ 5.6 × 10⁻³ | 0 (bins) |
| W-, decay-angle window | `"wm", recoil=["pt_w", "ctw_window"]` | 99.694% (99.694%) | 5.1 × 10⁻³; 0.040 | 0 |
| the same, windows A, B, C | `..., window=V` | 99.644, 99.697, 99.698% (the same) | ≤ 5.1 × 10⁻³ | ≤ 1.7 × 10⁻⁷ (bins) |
| the same, members 1, 29, 58 | `..., pdf_member=k` | 99.678, 99.575, 99.695% (the same) | ≤ 5.6 × 10⁻³ | 0 (bins) |
| gg → H | `"ggh"` | 88.148% (88.148%) | 2.1 × 10⁻⁵; 0.26 | 0 |
| gg → H, windows A, B, C, D | `"ggh", window=V` | 84.378, 91.086, 92.296, 91.810% (the same) | ≤ 2.4 × 10⁻⁴ | ≤ 2.5 × 10⁻¹⁰ (bins) |
| gg → H, CT18NNLO members 1, 29, 58 | `"ggh", pdf_member=k` | 88.190, 88.049, 88.277% (the same) | ≤ 4.2 × 10⁻⁵ | 0 (bins) |
| Z+jet, + 16 joint moments | `"zj"` | 92.588% (92.588%) | 7.2 × 10⁻³; 0.18 | 0 |
| Z+jet, + acoplanarity moments | `"zj", recoil=["ptj1", "ptj2", "aco"]` | 86.443% (86.443%) | 6.0 × 10⁻³; 0.17 | 0 |
| Z+jet, windows A, B, C | `"zj", window=V` | 91.966, 93.144, 92.581% (the same) | ≤ 3.8 × 10⁻³ | 0 (bins) |
| Z+jet, + acoplanarity, windows A, B, C | `"zj", recoil=[..., "aco"], window=V` | 85.799, 87.690, 86.541% (the same) | ≤ 5.3 × 10⁻³ | 0 (bins) |

These rows use the default `qt_basis="range"`: the p_T and jet towers on their acting range (a
window variant on the interval covering its own and the main tower's). The orders imposed are
m_ll 12, \|y_ll\| 11 and p_T^ℓℓ 16 for Drell-Yan, and m_ℓν 12, \|y_W\| 10 (W+) or 12 (W-) and
p_T^W 16 (W+) or 17 (W-) for W. For Z+jet they are m_ll 4, \|y_ll\| 5, p_T^j1 12, p_T^j2 6 and 4
acoplanarity orders. On the gg → H and merged diphoton priors the default criterion selects
every booked order (p_T^H 12, p_T^γγ 12). The window re-solves hold the central solve's orders from a cold start,
since a variant's polynomials can differ from the central solve's.

## The priors

| prior | events | used for |
|---|---|---|
| MG5_aMC LO pp → ℓ⁺ℓ⁻ + Pythia 8.312, 13 TeV, ATLAS fiducial selection on Born-level leptons | 747,028 | Drell-Yan |
| MG5_aMC LO pp → μ⁺ν, μ⁻ν̄ + Pythia 8.312, 7 TeV, the NNLOJET W selection | 521,316 (W+), 487,992 (W-) | W± |
| Pythia 8.312 gg → H, 13 TeV, inclusive | 2,000,000 | gg → H |
| merged MG5_aMC γγ + γγj (CKKW-L) with the Pythia gg box, nine slices, 8 TeV, ATLAS selection | 10,644,601 | diphoton (merged) |
| Pythia q q̄ → γγ and gg box, five slices, 8 TeV, ATLAS selection | 6,938,827 | diphoton (2 → 2) |
| MG5_aMC LO pp → ℓ⁺ℓ⁻ + Pythia 8.312, 13 TeV, parton-level jets, ≥ 1 jet | 934,906 | Z+jet |
| MG5_aMC LO pp → ℓ⁺ℓ⁻ (10⁷ events, mass window only) + Pythia 8.312, matrix-element corrections off, 13 TeV, ATLAS fiducial selection on Born-level leptons | 3,583,332 | Drell-Yan (the paper) |
| MG5_aMC LO pp → μν per charge + Pythia 8.312, matrix-element corrections off, 7 TeV, the NNLOJET W selection | 522,365 (W+), 490,046 (W-) | W± (the paper) |
| the same 10⁷ generated events as the paper's Drell-Yan prior, parton-level jets, ≥ 1 jet | 914,685 | Z+jet (the paper) |

The paper describes its priors (its appendix on samples), the letter its Drell-Yan prior. None
of the priors is distributed with the repository.
