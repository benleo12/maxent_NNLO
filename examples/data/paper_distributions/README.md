# Full-sample differential distributions

Each link opens a figure from the paper or its binned predictions. The runnable
examples fit compact event samples. These figures use the full production samples.

| Application | Figure | Binned predictions |
|---|---|---|
| Neutral-current Drell–Yan | [Dilepton mass, absolute rapidity and angular recoil](drell_yan/figure.pdf) | [Angular recoil in the ATLAS bins](drell_yan/atlas_phistar.csv), [generator spectra and NNLO references](drell_yan/distributions.csv) |
| Charged-current Drell–Yan | [Lepton charge asymmetry](w/figure.pdf), [hard-recoil acceptance check](w/acceptance.pdf) | [CSV](w/predictions.csv), [baseline data and residuals](w/baseline.csv), [scale members and covariances](w/charge_asymmetry.npz), [conditional acceptances](w/acceptance.csv) |
| Gluon-fusion Higgs | [Recoil and jets](higgs/jets.pdf), [diphoton recoil and rapidity after photon cuts](higgs/photons.pdf) | [Fiducial spectra, variations, inclusive recoil and jet counts](higgs/fiducial.npz) |
| Continuum diphotons | [Photon-pair mass and pseudorapidity separation](diphoton/figure.pdf) | [Densities, reference and finite-event uncertainties](diphoton/distributions.npz) |

The resolved-jet application has a [separate benchmark and refinement study](../zj_refinement/RESULTS.txt).

The [example catalogue](../../README.md) gives the event samples and fit inputs for each application.

## Reading the numbers

- **Drell–Yan.** The mass and rapidity panels compare the full MG5+Pythia and
  Herwig predictions with central NNLO. Mass uses the fiducial rate. Each
  rapidity shape has unit area for absolute dilepton rapidity below 2.4.
  The angular panel uses all 3,583,332 MG5+Pythia events and all 36 ATLAS
  bins. Their edges, measured Born-level densities, uncertainties, predictions,
  ratios and seven scale members are in `atlas_phistar.csv`. The two scale factors
  in each column heading multiply the dilepton mass. Each member includes the
  complete fiducial rate in its normalization. The prediction error treats
  fitted weights as fixed and includes its shared normalization. Data errors
  combine the published correlated and uncorrelated magnitudes in quadrature
  for each bin. The earlier generator-binned angular spectra remain in
  `distributions.csv`. [ATLAS Table 4b](https://doi.org/10.17182/hepdata.92377.v3/t5)
- **W.** The cuts in the three columns are 20, 28 and 32 GeV. The first comparison
  uses the combined ATLAS Born-lepton measurement. The other two use an independent NNLO
  integration. The NPZ includes the seven scale members, the separate covariances
  for integration and prior statistics, and differences between fits evaluated on
  the same events. The `models`, `pairs` and `scale_names` arrays give their
  order. The acceptance table supplies numerical errors on the paired change at fixed recoil. Both W plots have [editable TeX sources](w/source/README.md). [ATLAS Table XXVI](https://arxiv.org/pdf/1109.5141)
- **Higgs.** The fiducial spectra use absolute units, fb/GeV for transverse
  momenta and fb per unit rapidity. Only gluon fusion is reweighted. Each total
  includes the same VBF, WH, ZH and ttH contributions. The first leading-jet bin
  represents events without a selected jet above 30 GeV. The `metadata_json`
  field specifies the rate, scale, transition and finite-event uncertainties.
  The simulation applies isolation using the transverse energy of the summed track
  four-vector. The ATLAS paper specifies a scalar sum of track transverse momenta.
  The stored jets use visible final-state particles. The ATLAS particle-level
  definition treats prompt leptons and invisible decay products differently and
  removes jets close to selected electrons. These differences also enter the
  comparisons with matched generators.
  [ATLAS measurement](https://arxiv.org/abs/2202.00487) · [HEPData record](https://doi.org/10.17182/hepdata.137886.v1)
- **Continuum diphotons.** The file contains normalized densities for merged
  (`v6`) and shower-only (`v4`) events, before (`p`) and after (`q`) reweighting.
  The `m` arrays use photon-pair mass in GeV. The `eta` arrays use the
  hyperbolic tangent of half the absolute photon pseudorapidity difference.
  Ratio panels compare both reweighted samples to ATLAS in its published bins.
  Both samples and the fixed-order mass spectrum use smooth-cone isolation,
  with different radial profiles. ATLAS uses a fixed cone. [ATLAS mass spectrum](https://doi.org/10.17182/hepdata.77381.v1/t2) · [ATLAS angular spectrum](https://doi.org/10.17182/hepdata.77381.v1/t6)

All NPZ archives load with `numpy.load(path, allow_pickle=False)`.
File digests are listed in [SHA256.json](SHA256.json).
