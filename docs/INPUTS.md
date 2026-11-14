# Event inputs for the five applications

The same event-array interface serves every process. All five applications have
public moment tables, maps, windows, normalizations and fixed-order run cards.
You can fit your own compatible generator sample without any path from the
authors' machines. Historical production paths inside provenance records identify
the original calculations; the fitting commands do not read them.

## Choose an application

Every input is an NPZ file containing equal-length observable arrays and a
strictly positive `weight` array. Momenta and masses are in GeV; angles and
rapidities are dimensionless. Retain `event_id` and any additional analysis
observables in the same row order.

| Application | Arrays in addition to `weight` | Definition and fixed-order card |
|---|---|---|
| Drell–Yan, `dy` | `mll`, `y_abs`, `pT_ll` | 13 TeV; pre-QED lepton pair, 66<mll<116, both pT>27 and abs(eta)<2.5; [card](../runcards/dy_moments_log30_hi.run), [table](../moments/dy_nnlo_moments.json) |
| W±, `wp` / `wm` | `m_w`, `y_abs`, `pt_w`; `ctw1` for the angular or new joint examples | 7 TeV; charged pre-QED lepton and neutrino, pT_l>20, pT_nu>25, mT>40, abs(eta_l)<2.5; [W+ card](../runcards/w_moments_WP.run), [W− card](../runcards/w_moments_WM.run), [tables](../moments/README.md) |
| Higgs, `ggh` | `y_abs`, `pT_H` | 13 TeV inclusive on-shell Higgs, before decay acceptance cuts; [card](../runcards/ggh_moments3.run), [54 GeV window card](../runcards/ggh_winvar_cc.run), [table](../moments/ggh_nnlo_moments.json) |
| Diphotons, `aa` | `m_aa`, `abs_costh_aa`, `pt_aa` | 8 TeV; ordered photon pT>40/30, abs(eta)<2.37 excluding 1.37–1.56, photon separation>0.4 and the smooth-cone isolation in the [card](../runcards/gg_moments2.run); [table](../moments/aa_nnlo_moments.json) |
| Z+jet, `zj` | `mll`, `y_abs`, `ptj1`, `ptj2`; `pimdphi` for the acoplanarity example | Same 13 TeV DY leptons, plus at least one anti-kT R=0.4 jet with pT>20, abs(y)<4.4 and lepton–jet separation>0.4; [card](../runcards/zj_moments.run), [table](../moments/zj_nlo_moments.json) |

The cards specify the remaining electroweak, PDF, isolation and scale choices.
They describe the fixed-order calculation; they are not shower-generation cards.
The table's `fiducial`, `towers`, `joint` and `default_constraints` fields are the
complete fitting definition. Inspect the actual selected configuration before
running, for example:

```bash
python examples/paper_examples.py aa --inspect
python examples/paper_examples.py aa my_diphoton_prior.npz diphoton_weights.npz
```

Replace `aa` with `dy`, `wp`, `wm`, `ggh` or `zj`. The [example catalogue](../examples/README.md)
lists the optional angular, joint and acoplanarity configurations. Programmatic
`load_moments(process)` returns the complete `moments, config` pair; no private
configuration file is needed.

## Extract the arrays from event records

For a four-vector `(px, py, pz, E)`, use transverse momentum
`sqrt(px**2 + py**2)`, rapidity `0.5*log((E+pz)/(E-pz))` and invariant mass
`sqrt(E**2-px**2-py**2-pz**2)`. Pseudorapidity is `asinh(pz/pT)`.

- **DY:** sum the two signal lepton four-vectors before QED radiation. The pair
  supplies `mll`, `abs(y)` and `pT_ll`. The optional [HepMC adapter](HEPMC.md)
  performs this extraction and selection for the declared QED-off samples.
  Keep `abs_costh` and the smaller lepton momentum `pT_sub` for the real-DY
  example's independent angle and tighter-cut comparisons.
- **W:** sum the charged lepton and its neutrino. The pair supplies `m_w`,
  `abs(y)` and `pt_w`; use the charged-lepton sign convention in
  [the angle definition](../moments/README.md#what-your-prior-must-provide) for
  `ctw1 = 1 + cos(theta_CS)`. Form mT from
  `sqrt(2*pT_l*pT_nu*(1-cos(Delta_phi)))`. Keep signed `eta_l` and `pt_l` for
  lepton predictions. The [new W-study runner](../examples/w_joint_refinement.py)
  also shows this extraction explicitly for lepton/neutrino four-vector arrays.
- **Higgs:** take the on-shell Higgs four-vector to obtain `abs(y)` and `pT_H`.
  Fit the inclusive sample, then apply decay or analysis cuts using the resulting
  event weights. Retain the decay products, jet observables and branching-fraction
  convention needed for that later analysis.
- **Diphotons:** after the stated photon selection and isolation, sum the two
  photon four-vectors to obtain `m_aa` and `pt_aa`. Compute the absolute
  Collins–Soper angle with the same pair formula as DY, replacing leptons by
  photons. The precise formula is in the [moment guide](../moments/README.md#what-your-prior-must-provide).
  Preserve relative generation/slice weights when combining samples.
- **Z+jet:** construct the DY pair and the selected jets, then order the jets by
  pT. Set `ptj2=0` when there is no second selected jet. The paper clusters after
  shower and MPI but before hadronization. For the acoplanarity example,
  `pimdphi = pi - Delta_phi(lepton,lepton)`, with Delta_phi in [0,pi]. The
  [two-jet benchmark](../examples/data/zj_benchmark/README.md) additionally uses
  `njet` and applies its stated hard two-jet cuts.

The prior must cover the final analysis selection. Generation cuts that exclude
events which could enter that selection after showering can remove needed
configurations. Preserve original relative weights and any sampling corrections;
the public DY fixtures provide an explicit inverse-inclusion example.

## What runs with the checkout alone

The two [real DY fitting demonstrations](../examples/data/README.md) include
compact Pythia and Herwig event samples, fixed-order moments and reference arrays.
They perform new fits without an external event archive. The full-sample DY,
independent W and two-jet comparison plots use committed numerical caches.

Other application fits use your supplied compatible prior. The authors' complete
million-event productions are not prerequisites for using the released moments.
Reproducing their exact event-by-event results does require those same prior
events. For another process or observable, follow
[New processes and observables](NEW_PROCESSES.md): supply its shared definitions,
positive prior and compatible fixed-order moment inputs; the solver is unchanged.
