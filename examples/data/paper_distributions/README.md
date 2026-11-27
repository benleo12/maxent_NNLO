# Full-sample differential distributions

Each link opens a figure from the paper or its binned predictions. The runnable
examples fit compact event samples. These figures use the full production samples.

| Application | Figure | Binned predictions |
|---|---|---|
| Neutral-current Drell–Yan | [Dilepton recoil and lepton acoplanarity](drell_yan/recoil_and_phistar.pdf) | [Recoil in the ATLAS bins](drell_yan/atlas_recoil.csv), [both generators on common intervals](drell_yan/atlas_recoil_common_integrals.csv), [angular recoil in the ATLAS bins](drell_yan/atlas_phistar.csv), [generator spectra and NNLO references](drell_yan/distributions.csv), [acoplanarity against fixed order](drell_yan/phistar_fixed_order.md), [prior against fixed order](drell_yan/prior_phistar_fo.csv), [prior against data](drell_yan/prior_phistar_atlas.csv), [Herwig in the ATLAS acoplanarity bins](drell_yan/herwig_atlas_groups.md) |
| Charged-current Drell–Yan | [Lepton charge asymmetry](w/figure.pdf), [hard-recoil acceptance check](w/acceptance.pdf), [acceptance in recoil and polar angle bins](w/regional_acceptance.pdf) | [CSV](w/predictions.csv), [baseline data and residuals](w/baseline.csv), [original LO+PS charge rates](w/lo_ps_prior.csv), [regional acceptance differences](w/regional_acceptance.json), [scale members and covariances](w/charge_asymmetry.npz), [conditional acceptances](w/acceptance.csv) |
| Gluon-fusion Higgs | [Recoil and jets](higgs/jets.pdf), [diphoton recoil and rapidity after photon cuts](higgs/photons.pdf) | [Fiducial spectra, variations, inclusive recoil and jet counts](higgs/fiducial.npz), [fiducial fixed-order references](higgs/higgs_fixed_order.md), [inclusive recoil in the combined ATLAS bins](higgs/inclusive/production_recoil.md) |
| Continuum diphotons | [Recoil and azimuthal separation](diphoton/recoil_and_angle.pdf), [photon-pair mass and pseudorapidity separation](diphoton/figure.pdf) | [Recoil and azimuth in the ATLAS bins](diphoton/atlas_recoil_azimuth.md), [mass and angular densities](diphoton/distributions.npz), [mass and angular fixed-order bins](diphoton/mass_angle_reference.npz), [azimuth against fixed order](diphoton/diphoton_azimuth_fixed_order.md), [original priors](diphoton/recoil_angle_priors.csv), [measured-bin and grouped fitted ratios](diphoton/grouped_fit_ratios.csv) |
| $Z$+jet production | [Leading jet, jet balance and dilepton acoplanarity](zj/distributions.pdf), [two-jet momentum densities](zj/joint_jets.pdf) | [Independent two-jet benchmark and refinement study](../zj_refinement/RESULTS.txt), [two-dimensional bin probabilities](zj/joint_jets.csv) |

The [example catalogue](../../README.md) gives the event samples and fit inputs for each application.

## Reading the numbers

- **Drell–Yan.** The [supplementary mass and rapidity checks](drell_yan/mass_and_rapidity.pdf) compare the full MG5+Pythia and
  Herwig predictions with central NNLO. Mass uses the fiducial rate. Each
  rapidity shape has unit area for absolute dilepton rapidity below 2.4.
  The [earlier recoil and Collins–Soper angle comparison](drell_yan/recoil_and_angle.pdf) remains available.
  The acoplanarity comparison uses all 3,583,332 MG5+Pythia events and all 36 ATLAS
  bins. Their edges, measured Born-level densities, uncertainties, predictions,
  ratios and seven scale members are in `atlas_phistar.csv`. The two scale factors
  in each column heading multiply the dilepton mass. Each member includes the
  complete fiducial rate in its normalization. The prediction error treats
  fitted weights as fixed and includes its shared normalization. Data errors
  combine the published correlated and uncorrelated magnitudes in quadrature
  for each bin. The earlier generator-binned angular spectra remain in
  `distributions.csv`. [ATLAS Table 4b](https://doi.org/10.17182/hepdata.92377.v3/t5)
  The full Herwig sample contributes [four acoplanarity groups](drell_yan/herwig_atlas_groups.md)
  that compare with integrals of complete ATLAS bins.
  In `atlas_recoil.csv`, the same full event sample gives the recoil distribution
  before and after reweighting. The table lists the fiducial shape in GeV⁻¹ and
  the reweighted cross section at each scale in pb/GeV. The scale pairs
  (renormalization, factorization) are (1,1), (2,2), (½,½), (2,1), (1,2), (½,1)
  and (1,½), in that order. Finite-event errors include the shared fiducial
  normalization at fixed fitted weights. The two ATLAS error columns give the
  reported correlated and uncorrelated magnitudes in each bin. Only the
  prediction is supplied for 900–2500 GeV, where the HEPData table and the
  article differ by a factor equal to the bin width. [ATLAS Table 4a](https://doi.org/10.17182/hepdata.92377.v3/t4)
  The common-interval table gives fiducial event fractions for both showers
  and ATLAS on eight intervals with boundaries in all three original binnings.
  ATLAS supplies pointwise errors on its native intervals.
- **W.** The cuts in the three columns are 20, 28 and 32 GeV. Each cut has an independent NNLO comparison. The baseline
  also uses the combined ATLAS Born-lepton measurement. Its NNLO reference
  is in the [independent W validation archive](../w_joint/charge_asymmetry.json). The NPZ includes the seven scale members, the separate covariances
  for integration and prior statistics, and differences between fits evaluated on
  the same events. The `models`, `pairs` and `scale_names` arrays give their
  order. The acceptance table supplies numerical errors on the paired change at fixed recoil. Both W plots have [editable TeX sources](w/source/README.md). [ATLAS Table XXVI](https://arxiv.org/pdf/1109.5141)
- **Higgs.** Rates include gluon fusion, VBF, WH, ZH and ttH. Only gluon fusion is reweighted.
  The fiducial predictions use isolated prompt photons and particle-level jets. The first leading-jet bin contains the zero-jet cross section. Scale, transition, PDF and shower variations are supplied separately. Shower members use independent events with the nominal generator settings and each member is refitted.
  For a given spectrum and model, `_original_selection` gives the earlier result, `_selection_shift` gives the revised selection minus the original, and `_selection_shift_mc_error` is the Monte Carlo error on that difference, evaluated on the same events. Units, cuts and band definitions are in `metadata_json`.
  [ATLAS measurement](https://arxiv.org/abs/2202.00487) · [HEPData record](https://doi.org/10.17182/hepdata.137886.v2)
- **Continuum diphotons.** `distributions.npz` contains normalized densities for merged
  (`v6`) and shower-only (`v4`) events, before (`p`) and after (`q`) reweighting.
  The `m` arrays use photon-pair mass in GeV. The `eta` arrays use the
  hyperbolic tangent of half the absolute photon pseudorapidity difference.
  The middle rows compare with fixed order and the lower rows use ATLAS in its published bins. The [mass and angle reference](diphoton/mass_angle_reference.md) gives the orders, binning and normalization of the calculations.
  Both samples and the fixed-order mass spectrum use smooth-cone isolation,
  with different radial profiles. ATLAS uses a fixed cone. [ATLAS mass spectrum](https://doi.org/10.17182/hepdata.77381.v1/t2) · [ATLAS angular spectrum](https://doi.org/10.17182/hepdata.77381.v1/t6)

All NPZ archives load with `numpy.load(path, allow_pickle=False)`.
File digests are listed in [SHA256.json](SHA256.json).
